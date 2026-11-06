import argparse
import asyncio
import os
import time

from openai import AsyncOpenAI

from template_full_examples import *
# 导入新的优化模板
from utils_anchor_refine import init_anchor_from_fusion_runs
from utils_entropy_sampling import update_prediction_history  ## 2
from utils_example_selection import compress_prompt_examples, refresh_examples_with_anchor  ## 3
from utils_full_examples import *  ## 1

# 添加父目录到模块搜索路径，以便导入template和utils

random.seed(24)

# 每轮优化后指令中保留的样例数（KMeans 聚类簇数）；<=0 表示不做精简
# 为保证覆盖当前样例中的全部标签，最终数量可能略大于该值
EXAMPLE_NUM = 10

# 全局API并发信号量：限制同时请求的总数（覆盖全部 entity_pred、summarize、fuse_entities 等）
# 该信号量在 main() 中按 base_url 的并发能力初始化。
# 默认 60：留出余量避免触发服务端限流（5xx/429）
GLOBAL_API_SEM = asyncio.Semaphore(100)


def get_global_api_sem():
    """返回全局API信号量，未初始化时使用默认值。"""
    global GLOBAL_API_SEM
    return GLOBAL_API_SEM


def set_global_api_concurrency(max_concurrent):
    """根据服务端能力调整全局API并发上限。"""
    global GLOBAL_API_SEM
    new_size = max(1, int(max_concurrent))
    print(f"[GlobalSem] 调整全局API并发上限为 {new_size}")
    GLOBAL_API_SEM = asyncio.Semaphore(new_size)


def extract_json_block(text):
    """
    从文本中提取 JSON 内容，兼容以下情况：
    1. 标准格式：```json ... ```
    2. 缺少结尾 ```：```json ... <EOF>
    3. 缺少 json 标签：``` ... ```
    4. 无代码围栏：直接提取 JSON 数组或对象
    """
    # 1. 标准 ```json...```
    outs = re.findall(r'```json\s*([\s\S]*?)```', text, re.S)
    if outs:
        return outs
    # 2. 有 ```json 但缺少结尾 ```
    m = re.search(r'```json\s*([\s\S]+)', text, re.S)
    if m:
        return [m.group(1).strip()]
    # 3. 有 ``` 但缺少 json 标签
    outs = re.findall(r'```\s*([\s\S]*?)```', text, re.S)
    if outs:
        return outs
    # 4. 无围栏，直接匹配 JSON 数组或对象
    m = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', text, re.S)
    if m:
        return [m.group(1)]
    return []


def part2_post_process(refined_entities):
    # 优先提取结构化输出段落，避免将中英文分析过程嵌入新 prompt
    # 三个优化 prompt 均要求输出 "## Entity Definitions" + "## Entity Examples/Annotation Examples"
    m = re.search(
        r'(##\s*Entity Definitions\b[\s\S]*)',
        refined_entities, re.S
    )
    if m:
        return re.sub(r'\n{3,}', '\n\n', m.group(1)).strip()

    # 兜底：尝试 >>> Final Output <<< 分隔符（旧版模型可能输出）
    entity_list = refined_entities.split(">>> Final Output <<<")[-1].strip()

    # 清理英文和中文分析过程短语
    cleanup_patterns = [
        r'\(No verified examples found\)',
        r'\(Note:.*?\)',
        r'###?\s*Final Output\b',
        r'>>> Final Output <<<',
        r'Analysis:.*?(?=\n\d+\.|\n##|$)',
        r'Confusion Identification:.*?(?=\n\d+\.|\n##|$)',
        r'Cause Analysis:.*?(?=\n\d+\.|\n##|$)',
        r'Sub-semantic Analysis.*?(?=\n\d+\.|\n##|$)',
        r'Validation.*?(?=\n\d+\.|\n##|$)',
        r'Example Verification.*?(?=\n\d+\.|\n##|$)',
        # 中文分析过程短语
        r'分析[：:].*?(?=\n\d+\.|\n##|$)',
        r'混淆识别[：:].*?(?=\n\d+\.|\n##|$)',
        r'原因分析[：:].*?(?=\n\d+\.|\n##|$)',
        r'子语义分析.*?(?=\n\d+\.|\n##|$)',
        r'验证[：:].*?(?=\n\d+\.|\n##|$)',
        r'示例验证.*?(?=\n\d+\.|\n##|$)',
    ]

    for pattern in cleanup_patterns:
        entity_list = re.sub(pattern, '', entity_list, flags=re.DOTALL | re.IGNORECASE)

    entity_list = re.sub(r'\n{3,}', '\n\n', entity_list)
    return entity_list.strip()


def refine_instruction(refined_entities, prediction_history=None, anchor_data=None):
    """优化后指令的统一后处理：抽取结构化段落 -> 聚类精简样例（多样、低熵、覆盖全部标签）
    -> 用当前锚点集刷新所选样例的标注（anchor_data 为空时跳过）。"""
    prompt = compress_prompt_examples(part2_post_process(refined_entities), EXAMPLE_NUM,
                                      prediction_history=prediction_history)
    if anchor_data:
        prompt = refresh_examples_with_anchor(prompt, anchor_data)
    return prompt


async def entity_pred(ENTITY_TYPES, data, client, current_prompt, max_retries, batch_size=100, temp=0.0, seed=1234):
    """
    异步批量并发推理版本。
    batch_size: 控制并发数量（即同时请求API的数量），建议设置为 5-20 之间，取决于API的速率限制(RPM)。
    """

    # 中文数据集使用更大的 token 上限，避免 JSON 被截断

    async def process_single_item(index, item):
        text = item["text"]
        retries = 0
        predicted_entities = [{"text": text, "entities": []}]  # 默认空结果，防止出错时无返回
        while retries < max_retries:
            try:
                # 中文在 temperature=0 下极易触发复读机，首次也使用略高温度
                current_temp = temp if retries == 0 else 0.5
                # extra_params = {"repetition_penalty": 1.1} if is_chinese else {}
                response = await client.chat.completions.create(
                    model=MODELNAME,
                    seed=seed,
                    messages=[
                        {
                            "role": "user",
                            "content": "{}\n## Target Text：\n{}".format(current_prompt, text)
                        }
                    ],
                    temperature=current_temp,
                    max_tokens=None,
                )
                output_text = response.choices[0].message.content
                outs = extract_json_block(output_text)

                output_text = outs[-1] if outs else output_text
                entities_data = normalize_entities_output(json.loads(output_text))
                predicted_entities = [{"text": text, "entities": entities_data}]
                return index, predicted_entities
            except json.JSONDecodeError:
                print(output_text)
                print(f"JSON 解析失败，正在尝试重试 {retries + 1}/{max_retries}...")
                retries += 1  # 计数增加
                # JSON解析错误属于"内容错误"，短退避即可（1~2s + 抖动）
                backoff = 1.0 * (2 ** retries) + random.uniform(0, 0.5)
                await asyncio.sleep(backoff)

            except Exception as e:
                err_msg = str(e)
                # 区分瞬时性5xx/超时（需要更长退避）和其他错误
                is_server_err = (
                        "Internal Server Error" in err_msg
                        or "Service Unavailable" in err_msg
                        or "Bad Gateway" in err_msg
                        or "Gateway Timeout" in err_msg
                        or "Timeout" in err_msg
                        or "Connection" in err_msg
                        or "429" in err_msg
                )
                wait = 5.0 + retries * 3.0 if is_server_err else 1.0
                wait += random.uniform(0, 1.0)  # 抖动避免雪崩
                print(f"发生意外错误: {e} -> 退避 {wait:.1f}s 后重试 ({retries + 1}/{max_retries})")
                await asyncio.sleep(wait)
                retries += 1  # 计数增加
                # predicted_entities = []
        # 如果所有重试均失败，返回默认的空结果
        print(f"解析失败，样本索引 {index} 返回空结果。")
        return index, predicted_entities

    # --- 批量并发执行主逻辑 ---
    total_items = len(data)
    print(f"开始批量推理，数据总量: {total_items}, 并发数: {batch_size}")

    tasks = [process_single_item(i, item) for i, item in enumerate(data)]

    local_semaphore = asyncio.Semaphore(batch_size)
    global_semaphore = get_global_api_sem()

    async def bounded_task(task):
        # 双重信号量：先取全局并发额度，再取本次调用的局部额度
        # 这样多个 entity_pred / summarize 之间的总并发永远不会超过 GLOBAL_API_SEM
        async with global_semaphore:
            async with local_semaphore:
                return await task

    results = await asyncio.gather(*[bounded_task(task) for task in tasks])

    y_pred = []
    for _, preds in results:
        y_pred.extend(preds)

    # 计算指标（保持原有逻辑）
    metrics = calculate_metrics_all(ENTITY_TYPES, data, y_pred)
    return y_pred, metrics


async def compare(ENTITY_TYPES, data, update_data, client, max_retries, current_entity, current_prompt, pred_errors,
                  meta=False, prev_y_pred=None, prediction_history=None, global_prediction_pool=None):
    if meta:  # 第一轮初始化，直接用现有指令预测
        # 注入角色定位：精确性专家
        # role_prompt = current_prompt + "\n\n## Expert Role: Precision Specialist\nFocus on accurate entity boundaries and correct entity type classification."
        y_pred, metrics = await entity_pred(ENTITY_TYPES, data, client, current_prompt, max_retries, temp=0.0,
                                            seed=1234)
        return y_pred, current_entity, metrics
    # 正则获取实体列表
    match = re.search(r"(?<=entities list:)(.*?)(?=notes:)", current_entity, re.I | re.S)
    current_entity = match.group(1).strip() if match else ""
    if not current_entity:
        current_entity = ENTITY_TYPES
    # 采样错误情况用于分析和改写专家指令
    print("compare-predict-errors:", pred_errors)
    full_examples_str = get_full_entity_examples_from_errors(update_data, pred_errors, max_samples=10)
    analy_error_prompt = PROMPT_compare_1.format(current_entity, full_examples_str, ENTITY_TYPES)
    if pred_errors:
        try:
            response1 = await client.chat.completions.create(
                model=MODELNAME,
                seed=1234,
                messages=[
                    {"role": "user", "content": analy_error_prompt},
                ],
                temperature=0.0,
            )
            # 指令修改
            refined_entities = response1.choices[0].message.content
            new_prompt = Expert_prompt_template.format(
                refine_instruction(refined_entities, global_prediction_pool or prediction_history,
                                   anchor_data=update_data)
            )

        except Exception as e:
            print("发生错误，对比-1")
            new_prompt = current_prompt
    else:
        print("compare: 错误分析未发现错误样本")
        new_prompt = current_prompt
    print("compare-prompt-after", new_prompt)
    # 优化2：智能跳过 - prompt未变时复用上轮结果
    if new_prompt == current_prompt and prev_y_pred is not None:
        print("   >>> [优化] Compare-Agent: prompt未变，跳过重新预测，复用上轮结果")
        metrics = calculate_metrics_all(ENTITY_TYPES, data, prev_y_pred)
        return prev_y_pred, new_prompt, metrics
    y_pred, metrics = await entity_pred(ENTITY_TYPES, data, client, new_prompt, max_retries, temp=0.0)
    print("compare-训练集上测试:", metrics)
    return y_pred, new_prompt, metrics


async def parsing(ENTITY_TYPES, data, update_data, client, max_retries, current_entity, current_prompt, recall_errors,
                  meta=False, prev_y_pred=None, prediction_history=None, global_prediction_pool=None):
    if meta:  # 第一轮初始化，直接用现有指令预测
        # 注入角色定位：召回性专家
        # role_prompt = current_prompt + "\n\n## Expert Role: Recall Specialist\nIdentify as many entities as possible while maintaining accuracy."
        y_pred, metrics = await entity_pred(ENTITY_TYPES, data, client, current_prompt, max_retries, temp=0.0,
                                            seed=1234)
        return y_pred, current_prompt, metrics
    match = re.search(r"(?<=entities list:)(.*?)(?=notes:)", current_entity, re.I | re.S)
    current_entity = match.group(1).strip() if match else ""
    if not current_entity:
        current_entity = ENTITY_TYPES
    # 采样错误，归纳经验改写指令
    full_examples_str = get_full_entity_examples_from_errors(update_data, recall_errors, max_samples=10)
    first_round_prompt = PROMPT_parse_1.format(current_entity, full_examples_str, ENTITY_TYPES)
    if recall_errors:
        try:
            first_response = await client.chat.completions.create(
                model=MODELNAME,
                seed=1234,
                messages=[
                    {"role": "user", "content": first_round_prompt}
                ],
                temperature=0.0,
            )

            refined_entities = first_response.choices[0].message.content
            new_prompt = Expert_prompt_template.format(
                refine_instruction(refined_entities, global_prediction_pool or prediction_history,
                                   anchor_data=update_data)
            )
        except Exception as e:
            print("发生错误，解析-1")
            new_prompt = current_prompt
    else:
        print("parsing: 错误分析未发现错误。")
        new_prompt = current_prompt
    print("parsing-prompt-after", new_prompt)
    if new_prompt == current_prompt and prev_y_pred is not None:
        print("   >>> [优化] Parsing-Agent: prompt未变，跳过重新预测，复用上轮结果")
        metrics = calculate_metrics_all(ENTITY_TYPES, data, prev_y_pred)
        return prev_y_pred, new_prompt, metrics
    y_pred, metrics = await entity_pred(ENTITY_TYPES, data, client, new_prompt, max_retries, temp=0.0)
    print("parsing-训练集上测试:", metrics)
    return y_pred, new_prompt, metrics


async def example(ENTITY_TYPES, data, update_data, client, max_retries, current_entity, current_prompt, pred_errors,
                  meta=False, prev_y_pred=None, prediction_history=None, global_prediction_pool=None):
    if meta:  # 第一轮初始化，直接用现有指令预测
        y_pred, metrics = await entity_pred(ENTITY_TYPES, data, client, current_prompt, max_retries, temp=0.0,
                                            seed=1234)
        return y_pred, current_prompt, metrics
    match = re.search(r"(?<=entities list:)(.*?)(?=notes:)", current_entity, re.I | re.S)
    current_entity = match.group(1).strip() if match else ""
    if not current_entity:
        current_entity = ENTITY_TYPES
    # 还原完整正确标注，不附带错误上下文（example 专家只需要正确示例）
    full_examples_str = get_full_entity_examples_from_errors(update_data, pred_errors, max_samples=10,
                                                             with_error_context=False)
    first_step = PROMPT_example_1.format(current_entity, full_examples_str, ENTITY_TYPES)
    if pred_errors:
        try:
            response1 = await client.chat.completions.create(
                model=MODELNAME,
                seed=1234,
                messages=[
                    {"role": "user", "content": first_step},
                ],
                temperature=0.0,
            )
            # 指令修改
            refined_entities = response1.choices[0].message.content
            new_prompt = Expert_prompt_template.format(
                refine_instruction(refined_entities, global_prediction_pool or prediction_history,
                                   anchor_data=update_data)
            )
        except Exception as e:
            print("发生错误，示例-1")
            new_prompt = current_prompt
    else:
        print("example:错误分析未发现错误")
        new_prompt = current_prompt

    print("example-prompt-after", new_prompt)
    # 优化2：智能跳过 - prompt未变时复用上轮结果
    if new_prompt == current_prompt and prev_y_pred is not None:
        print("   >>> [优化] Example-Agent: prompt未变，跳过重新预测，复用上轮结果")
        metrics = calculate_metrics_all(ENTITY_TYPES, data, prev_y_pred)
        return prev_y_pred, new_prompt, metrics
    y_pred, metrics = await entity_pred(ENTITY_TYPES, data, client, new_prompt, max_retries, temp=0.0)
    print("example-在训练集上测试:", metrics)
    return y_pred, new_prompt, metrics


async def fuse_entities(max_retries, client, agent1, entities1, agent2, entities2, agent3, entities3, text,
                        temp=0.0, stage_train=True):
    match = re.search(r"(?<=entities list:)(.*?)(?=notes:)", agent1, re.I | re.S)
    agent1 = match.group(1).strip() if match else ""
    match = re.search(r"(?<=entities list:)(.*?)(?=notes:)", agent2, re.I | re.S)
    agent2 = match.group(1).strip() if match else ""
    match = re.search(r"(?<=entities list:)(.*?)(?=notes:)", agent3, re.I | re.S)
    agent3 = match.group(1).strip() if match else ""

    if stage_train:
        system_content = Fusion_system_train.format(agent1, entities1, agent2, entities2, agent3, entities3)
        user_content = Fusion_user_train.format(text)
    else:
        # 使用纯指令（不包含专家预测结果）
        system_content = Fusion_system_test.format(agent1, agent2, agent3)
        user_content = Fusion_user_test.format(text)

    # 第一步：纯指令推理
    pure_instruction_entities = []
    retries = 0
    global_semaphore = get_global_api_sem()
    while retries < max_retries:
        try:
            async with global_semaphore:
                response = await client.chat.completions.create(
                    model=MODELNAME,
                    seed=1234,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temp,
                    max_tokens=None
                )

            output_text = response.choices[0].message.content
            try:
                outs = extract_json_block(output_text)
                output_text = outs[-1] if outs else output_text
                pure_instruction_entities = normalize_entities_output(json.loads(output_text))
            except json.JSONDecodeError:
                pure_instruction_entities = []
            break
        except Exception as e:
            retries += 1
            print(f"纯指令推理失败，重试({retries})：{e}")
            await asyncio.sleep(10)

    # 第二步：三个专家的预测结果做 2/3 共识投票
    entity_votes = {}  # {(type, entity): (entity_obj, vote_count)}

    for entities in [entities1, entities2, entities3]:
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                    key = (ent["type"].strip().lower(), ent["entity"].strip())
                    if key not in entity_votes:
                        entity_votes[key] = (ent, 1)
                    else:
                        obj, count = entity_votes[key]
                        entity_votes[key] = (obj, count + 1)

    # 保留获得至少 1 票的实体（降低门槛，避免丢失单专家预测的正确实体）
    consensus_entities = [obj for (_, _), (obj, count) in entity_votes.items() if count >= 1]

    # 第三步：整合纯指令推理结果和共识投票结果
    # 使用字典记录所有候选实体及其投票数
    integrated_dict = {}

    # 添加纯指令推理的结果（降低权重，让专家投票主导）
    for ent in pure_instruction_entities:
        if isinstance(ent, dict) and "type" in ent and "entity" in ent:
            key = (ent["type"].strip().lower(), ent["entity"].strip())
            integrated_dict[key] = (ent, 1)  # 纯指令推理结果权重降为1（原为3）

    # 添加共识投票结果
    for ent in consensus_entities:
        if isinstance(ent, dict) and "type" in ent and "entity" in ent:
            key = (ent["type"].strip().lower(), ent["entity"].strip())
            if key not in integrated_dict:
                vote_count = entity_votes.get(key, (ent, 0))[1]
                integrated_dict[key] = (ent, vote_count)

    # 第四步：使用类似 resolve_overlapping_entities 的方法去重和解决冲突
    final_entities = resolve_overlapping_entities(integrated_dict, text)

    return final_entities


async def process_single_item(item, agent1_out, agent2_out, agent3_out, client, max_retries, agent1, agent2, agent3,
                              temp=0.0, stage_train=True):
    text = item["text"]
    sum_entities = await fuse_entities(max_retries, client, agent1, agent1_out, agent2, agent2_out, agent3,
                                       agent3_out, text, temp=temp, stage_train=stage_train)
    return {"text": text, "entities": sum_entities}


async def summarize(ENTITY_TYPES, datas, client, max_retries, agent1, agent1_outs, agent2, agent2_outs,
                    agent3, agent3_outs, batch_size=100, temp=0.0, stage_train=True):
    print(f"开始批量推理融合，数据总量: {len(datas)}，并发批次大小 (Batch Size): {batch_size}")

    semaphore = asyncio.Semaphore(batch_size)

    async def bounded_process(item, agent1_out, agent2_out, agent3_out):
        async with semaphore:
            return await process_single_item(item, agent1_out, agent2_out, agent3_out, client, max_retries,
                                             agent1, agent2, agent3, temp=temp, stage_train=stage_train)

    tasks = [
        bounded_process(item, agent1_out, agent2_out, agent3_out)
        for item, agent1_out, agent2_out, agent3_out in zip(datas, agent1_outs, agent2_outs, agent3_outs)
    ]

    y_pred = await asyncio.gather(*tasks)

    metrics = calculate_metrics_all(ENTITY_TYPES, datas, y_pred)
    return y_pred, metrics


def print_evl_metrict(metrics_example, iteration, agent_name):
    print(">>>>> {}-第{}轮评估".format(agent_name, iteration + 1))
    print("Macro-F1:", metrics_example["macro_f1"])
    print("Macro-Precision:", metrics_example["macro_precision"])
    print("Macro-Recall:", metrics_example["macro_recall"])
    print("Micro-F1:", metrics_example["micro_f1"])
    print("Micro-Precision:", metrics_example["micro_precision"])
    print("Micro-Recall:", metrics_example["micro_recall"])
    print(metrics_example["details"])


async def expert_fusion(ENTITY_TYPES, test_data, client, max_retries, original_prompt1, entities1,
                        original_prompt2, entities2, original_prompt3, entities3, dataset, iteration, num_index,
                        model_name, small_diff_threshold, drop_threshold, if_log=False, temp=0.0, stage_train=True):
    print(len(test_data), len(entities1), len(entities2), len(entities3))

    y_pred, metrics = await summarize(ENTITY_TYPES, test_data, client, max_retries, original_prompt1,
                                      entities1, original_prompt2, entities2, original_prompt3, entities3, temp=temp,
                                      stage_train=stage_train)

    text_to_entities = defaultdict(list)
    for item in y_pred:
        text = item["text"]
        text_to_entities[text].extend(item.get("entities", []))

    merged_result = []
    for text, entities in text_to_entities.items():
        merged_result.append({
            "text": text,
            "entities": entities
        })
    if if_log:
        metrics_all = calculate_metrics_all(ENTITY_TYPES, test_data, merged_result)
        print_evl_metrict(metrics_all, 0, "最终融合结果")
        print("保存最终评估结果：")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = f"./main_results/{model_name}/{dataset}/sdt{small_diff_threshold}_dt{drop_threshold}"
        os.makedirs(result_dir, exist_ok=True)
        result_path = f"{result_dir}/{timestamp}_iter{iteration + 1}_run{num_index}.json"

        json.dump(metrics_all, open(result_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=4)
        print(f"结果已保存至: {result_path}")
    return merged_result


def evaluate_against_anchors(y_pred, anchor_data, ENTITY_TYPES):
    """
    计算预测结果相对于锚点集（伪标签）的性能指标和实体数量
    使用 Macro-F1：对每个实体类型分别计算 F1，然后求平均
    """
    # 按类型分组构建锚点集和预测集
    anchor_by_type = {etype: set() for etype in ENTITY_TYPES}
    for item in anchor_data:
        text = item["text"]
        for ent in item.get("entities", []):
            etype = ent["type"].strip().lower()
            if etype in anchor_by_type:
                anchor_by_type[etype].add((text, ent["entity"].strip()))

    pred_by_type = {etype: set() for etype in ENTITY_TYPES}
    for item in y_pred:
        text = item["text"]
        for ent in item.get("entities", []):
            if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                etype = ent["type"].strip().lower()
                if etype in pred_by_type:
                    pred_by_type[etype].add((text, ent["entity"].strip()))

    # 对每个类型分别计算 F1
    f1_scores = []
    precision_scores = []
    recall_scores = []
    total_pred_count = 0

    for etype in ENTITY_TYPES:
        anchor_set = anchor_by_type[etype]
        pred_set = pred_by_type[etype]

        tp = len(pred_set & anchor_set)
        fp = len(pred_set - anchor_set)
        fn = len(anchor_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        total_pred_count += len(pred_set)

    # Macro-F1: 对所有类型的 F1 求平均
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    macro_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    macro_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    return {
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "count": total_pred_count
    }


def evaluate_pseudo_labels(pseudo_labels, gold_data, ENTITY_TYPES):
    """
    计算伪标签相对于金标准的质量指标
    使用与 calculate_metrics_all 相同的逻辑：逐样本匹配，然后按类型聚合统计

    Args:
        pseudo_labels: 伪标签数据 S_pseudo^(t)
        gold_data: 金标准数据 E_gold
        ENTITY_TYPES: 实体类型列表

    Returns:
        dict: 包含以下指标
            - pseudo_label_precision: 伪标签精确率（Macro平均）
            - pseudo_label_recall: 伪标签召回率（Macro平均）
            - pseudo_label_f1: 伪标签F1值（Macro-F1）
            - false_positive_count: FP数量 |S_pseudo^(t) \\ E_gold|
            - false_positive_rate: FP率 |S_pseudo^(t) \\ E_gold| / |S_pseudo^(t)|
            - pseudo_label_error_rate: 错误率（与FPR相同）
            - num_pseudo_entities: 伪标签实体总数
            - num_gold_entities: 金标准实体总数
    """
    # 使用与 calculate_metrics_all 相同的逻辑
    target_types = set(t.lower() for t in ENTITY_TYPES)
    type_stats = {t: {'TP': 0, 'FP': 0, 'FN': 0} for t in target_types}

    # 将伪标签转换为字典，以文本内容为键
    pseudo_dict = {item["text"]: item for item in pseudo_labels}

    # 逐样本遍历金标准数据
    for gold_item in gold_data:
        text = gold_item["text"]

        # 获取当前文本的真实实体集合 {(type, entity), ...}
        gold_entities = set()
        for ent in gold_item.get("entities", []):
            if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                t_type = ent["type"].strip().lower()
                if t_type in target_types:
                    gold_entities.add((t_type, ent["entity"].strip()))

        # 获取当前文本的伪标签实体集合
        pseudo_entities = set()
        if text in pseudo_dict:
            for ent in pseudo_dict[text].get("entities", []):
                if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                    t_type = ent["type"].strip().lower()
                    if t_type in target_types:
                        pseudo_entities.add((t_type, ent["entity"].strip()))

        # 计算当前样本中每个类别的 TP, FP, FN
        # TP: 预测正确 (交集)
        for t, _ in gold_entities & pseudo_entities:
            type_stats[t]['TP'] += 1

        # FP: 预测了但不在真实集中 (预测集 - 真实集)
        for t, _ in pseudo_entities - gold_entities:
            type_stats[t]['FP'] += 1

        # FN: 真实存在但未被预测 (真实集 - 预测集)
        for t, _ in gold_entities - pseudo_entities:
            type_stats[t]['FN'] += 1

    # 对每个类型分别计算指标
    f1_scores = []
    precision_scores = []
    recall_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for etype in ENTITY_TYPES:
        etype_lower = etype.lower()
        tp = type_stats[etype_lower]['TP']
        fp = type_stats[etype_lower]['FP']
        fn = type_stats[etype_lower]['FN']

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Macro平均
    macro_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    macro_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # 全局统计（用于 FP 相关指标）
    num_pseudo_entities = total_tp + total_fp
    num_gold_entities = total_tp + total_fn

    # FP相关指标
    false_positive_count = total_fp  # N_FP^(t) = |S_pseudo^(t) \ E_gold|
    false_positive_rate = total_fp / num_pseudo_entities if num_pseudo_entities > 0 else 0.0  # R_FP^(t)

    # 错误率（与FPR相同）
    pseudo_label_error_rate = false_positive_rate

    return {
        "pseudo_label_precision": macro_precision,
        "pseudo_label_recall": macro_recall,
        "pseudo_label_f1": macro_f1,
        "false_positive_count": false_positive_count,
        "false_positive_rate": false_positive_rate,
        "pseudo_label_error_rate": pseudo_label_error_rate,
        "num_pseudo_entities": num_pseudo_entities,
        "num_gold_entities": num_gold_entities,
        "num_correct_entities": total_tp,
        "num_incorrect_entities": total_fp,
        "num_missing_entities": total_fn,
        "details": type_stats  # 添加详细统计，便于调试
    }


def compute_fusion_change(prev_merged, curr_merged):
    """计算两次融合结果之间的实体集合变化率（Jaccard距离）"""

    def to_entity_set(merged):
        s = set()
        for item in merged:
            text = item["text"]
            for ent in item.get("entities", []):
                if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                    s.add((text, ent["type"].strip().lower(), ent["entity"].strip()))
        return s

    prev_set = to_entity_set(prev_merged)
    curr_set = to_entity_set(curr_merged)
    union = prev_set | curr_set
    if not union:
        return 0.0
    return len(prev_set ^ curr_set) / len(union)


def has_overlap(entity1, entity2, text):
    """
    基于文本中实际位置判断重叠

    Args:
        entity1: 第一个实体文本
        entity2: 第二个实体文本
        text: 原始文本

    Returns:
        True 如果两个实体在文本中有位置重叠，否则 False
    """
    # 标准化
    e1_lower = entity1.strip().lower()
    e2_lower = entity2.strip().lower()
    text_lower = text.lower()

    # 如果完全相同，算重叠
    if e1_lower == e2_lower:
        return True

    def is_word_boundary(text, pos):
        """检查位置是否是词边界（前后是空格、标点或文本边界）"""
        if pos < 0 or pos > len(text):
            return False
        if pos == 0 or pos == len(text):
            return True
        # 检查该位置前后的字符
        before = text[pos - 1] if pos > 0 else ' '
        after = text[pos] if pos < len(text) else ' '
        # 词边界：空格、标点、特殊字符
        boundary_chars = ' \t\n\r.,;:!?()[]{}"\'-/'
        return before in boundary_chars or after in boundary_chars

    # 找到 entity1 在文本中的所有位置（考虑词边界）
    positions1 = []
    start = 0
    while True:
        pos = text_lower.find(e1_lower, start)
        if pos == -1:
            break
        # 检查词边界：实体前后都应该是词边界
        if is_word_boundary(text_lower, pos) and is_word_boundary(text_lower, pos + len(e1_lower)):
            positions1.append((pos, pos + len(e1_lower)))
        start = pos + 1

    # 找到 entity2 在文本中的所有位置（考虑词边界）
    positions2 = []
    start = 0
    while True:
        pos = text_lower.find(e2_lower, start)
        if pos == -1:
            break
        if is_word_boundary(text_lower, pos) and is_word_boundary(text_lower, pos + len(e2_lower)):
            positions2.append((pos, pos + len(e2_lower)))
        start = pos + 1

    # 如果任一实体在文本中找不到（考虑词边界后），不算重叠
    if not positions1 or not positions2:
        return False

    # 检查是否有任何位置对存在重叠
    for s1, e1 in positions1:
        for s2, e2 in positions2:
            # 区间重叠：不满足"一个完全在另一个之前"
            if not (e1 <= s2 or e2 <= s1):
                return True

    return False


def resolve_overlapping_entities(entities_dict, text):
    """
    解决重叠实体的冲突，策略：
    1. 相同标签 + 重叠边界：保留较长的实体（更完整）
    2. 不同标签 + 重叠边界：保留投票数更多的，投票数相同则保留较长的

    Args:
        entities_dict: {(type, entity): (entity_obj, vote_count)} 格式的字典
        text: 原始文本

    Returns:
        去重后的实体列表
    """
    if not entities_dict:
        return []

    # 转换为列表便于处理
    entity_list = [(type_label, entity_text, obj, votes)
                   for (type_label, entity_text), (obj, votes) in entities_dict.items()]
    #
    # # 按投票数降序排序，投票数相同则按实体长度降序
    # entity_list.sort(key=lambda x: (x[3], len(x[1])), reverse=True)

    # 贪心：按（票数, 长度）从高到低依次入选，与任一已选实体重叠者被丢弃
    # 相同标签重叠、不同标签重叠统一按此规则裁决
    kept = []
    for type_label, entity_text, obj, votes in entity_list:
        # if any(has_overlap(entity_text, k_text, text) for _, k_text, _ in kept):
        #     continue
        kept.append((type_label, entity_text, obj))

    return [obj for _, _, obj in kept]


def conservative_anchor_update(prev_anchor, new_merged, expert_preds_list, consensus_threshold=2,
                               fusion_vote_weight=1.0):
    """
    保守锚点更新策略，实现公式：
        S_pseudo^(t) = C^(t) ∪ (E_int^(t) ∩ S_pseudo^(t-1))

    其中：
        C^(t)              — 高共识实体集合（被 >= consensus_threshold 票的实体，
                              fusion_vote_weight > 0 时融合结果 new_merged 作为第四个投票源
                              按该权重计入票数，使三专家都未发现、但融合综合三份专家定义后
                              识别出的实体也能进入锚点；默认 0 即旧行为，只看专家票数）
        E_int^(t)          — 融合模型预测的实体集合（new_merged）
        S_pseudo^(t-1)     — 上一轮伪标签锚点集（prev_anchor）

    新增：处理边界重叠的实体
        - 相同标签 + 重叠边界：保留较长的实体（更完整）或投票数更多的
        - 不同标签 + 重叠边界：保留投票数更多的，投票数相同则保留较长的

    Args:
        prev_anchor: 上一轮的锚点集 S_pseudo^(t-1)
        new_merged: 本轮融合结果 E_int^(t)
        expert_preds_list: [expert1_preds, expert2_preds, expert3_preds] 各专家预测列表
        consensus_threshold: 至少需要多少票才算高共识（默认2）
        fusion_vote_weight: 融合结果计入 C^(t) 投票的票数权重（默认 2.0）

    Returns:
        更新后的锚点集 S_pseudo^(t)
    """
    # 构建专家预测的快速查找字典
    expert_dicts = []
    for preds in expert_preds_list:
        d = {}
        for item in preds:
            d[item["text"]] = item.get("entities", [])
        expert_dicts.append(d)

    # 构建融合结果（E_int^(t)）的快速查找字典
    new_merged_dict = {}
    for item in new_merged:
        new_merged_dict[item["text"]] = item.get("entities", [])

    # 构建上一轮锚点（S_pseudo^(t-1)）的快速查找字典
    prev_anchor_dict = {}
    for item in prev_anchor:
        prev_anchor_dict[item["text"]] = item.get("entities", [])

    # 收集所有文本（取并集，确保覆盖所有样本）
    all_texts = set(prev_anchor_dict.keys()) | set(new_merged_dict.keys())

    new_anchor = []
    for text in all_texts:
        # 统计每个实体被多少专家预测到，构建 C^(t)
        # 使用字典记录每个实体的投票数和对象：{(type, entity): (entity_obj, vote_count)}
        entity_votes_with_obj = {}

        for ed in expert_dicts:
            if text in ed:
                for ent in ed[text]:
                    if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                        key = (ent["type"].strip().lower(), ent["entity"].strip())
                        if key not in entity_votes_with_obj:
                            entity_votes_with_obj[key] = (ent, 1)
                        else:
                            obj, count = entity_votes_with_obj[key]
                            entity_votes_with_obj[key] = (obj, count + 1)

        # 融合结果作为第四个投票源
        if fusion_vote_weight:
            for ent in new_merged_dict.get(text, []):
                if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                    key = (ent["type"].strip().lower(), ent["entity"].strip())
                    if key not in entity_votes_with_obj:
                        entity_votes_with_obj[key] = (ent, fusion_vote_weight)
                    else:
                        obj, count = entity_votes_with_obj[key]
                        entity_votes_with_obj[key] = (obj, count + fusion_vote_weight)

        # C^(t)：高共识实体集合（投票数 >= consensus_threshold）
        high_consensus_dict = {k: v for k, v in entity_votes_with_obj.items()
                               if v[1] >= consensus_threshold}

        # E_int^(t)：融合模型预测的实体集合
        integrated_entities = new_merged_dict.get(text, [])
        integrated_dict = {}
        for ent in integrated_entities:
            if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                key = (ent["type"].strip().lower(), ent["entity"].strip())
                # 获取该实体的投票数（如果有的话）
                vote_count = entity_votes_with_obj.get(key, (ent, 0))[1]
                integrated_dict[key] = (ent, vote_count)

        # S_pseudo^(t-1)：上一轮锚点实体集合
        prev_entities = prev_anchor_dict.get(text, [])
        prev_dict = {}
        for ent in prev_entities:
            if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                key = (ent["type"].strip().lower(), ent["entity"].strip())
                # 获取该实体的投票数（如果有的话）
                vote_count = entity_votes_with_obj.get(key, (ent, 0))[1]
                prev_dict[key] = (ent, vote_count)

        # 公式：S_pseudo^(t) = C^(t) ∪ (E_int^(t) ∩ S_pseudo^(t-1))
        # 收集所有候选实体
        candidate_entities_dict = {}

        # 第一项：C^(t) — 高共识实体
        candidate_entities_dict.update(high_consensus_dict)

        # 第二项：E_int^(t) ∩ S_pseudo^(t-1) — 融合结果与旧锚点的交集
        for key in integrated_dict:
            if key in prev_dict:
                # 保留投票数更高的版本
                int_obj, int_votes = integrated_dict[key]
                prev_obj, prev_votes = prev_dict[key]
                if key not in candidate_entities_dict:
                    # 选择投票数更高的
                    candidate_entities_dict[key] = integrated_dict[key] if int_votes >= prev_votes else prev_dict[key]

        # 处理重叠实体冲突
        final_entities = resolve_overlapping_entities(candidate_entities_dict, text)

        new_anchor.append({"text": text, "entities": final_entities})

    return new_anchor


def update_expert_with_safeguard(agent_name, prev_state, new_prompt, new_y_pred, anchor_data, ENTITY_TYPES, iteration=0,
                                 exploration_rounds=4, small_diff_threshold=0.01, drop_threshold=0.03,
                                 large_drop_patience=3, logger=print):
    """
    接受专家更新，并追踪F1微小变化次数用于收敛判断。

    方案A核心：每次都用当前anchor集重新评估新旧预测，保证公平比较。
    - 删除metrics缓存，避免用不同anchor集的F1值比较
    - 在同一anchor集下比较新旧预测，单调性保证语义清晰
    - anchor集变化时，旧预测会被重新评估，如果表现不佳会被新预测替换

    探索机制：前 exploration_rounds 轮不做单调性保护，允许自由探索。
    """
    small_diff_count = prev_state.get("small_diff_count", 0)
    large_drop_count = prev_state.get("large_drop_count", 0)

    # 关键改动：删除缓存逻辑，每次都用当前anchor集重新评估新旧预测
    prev_metrics = evaluate_against_anchors(prev_state["y_pred"], anchor_data, ENTITY_TYPES)
    new_metrics = evaluate_against_anchors(new_y_pred, anchor_data, ENTITY_TYPES)

    old_f1 = prev_metrics["f1"]
    new_f1 = new_metrics["f1"]
    f1_diff = new_f1 - old_f1

    logger(f"   >>> Check {agent_name}: F1 {old_f1:.4f}->{new_f1:.4f} (diff: {f1_diff:.4f})")

    if f1_diff < -drop_threshold:
        large_drop_count += 1
        small_diff_count += 1  # 大幅下降也计入未改善次数，加速收敛判断
        logger(
            f"   >>> 大幅下降丢弃: {agent_name} F1 下降 {f1_diff:.4f} 超过阈值 -{drop_threshold}，保留旧模板，大幅波动计数: {large_drop_count}/{large_drop_patience}")
        return False, {
            "prompt": prev_state["prompt"],
            "y_pred": prev_state["y_pred"],
            "small_diff_count": small_diff_count,
            "large_drop_count": large_drop_count
        }
    else:
        large_drop_count = 0

    # if f1_diff < -small_diff_threshold:
    #     # 单调性保护：小幅下降同样拒绝，确保 prev_state 始终持有历史最优
    #     small_diff_count += 1
    #     logger(
    #         f"   >>> 单调性保护: {agent_name} F1 下降 {f1_diff:.4f}，保留旧模板，连续未改善计数: {small_diff_count}")
    #     return False, {
    #         "prompt": prev_state["prompt"],
    #         "y_pred": prev_state["y_pred"],
    #         "small_diff_count": small_diff_count,
    #         "large_drop_count": large_drop_count
    #     }

    if abs(f1_diff) <= small_diff_threshold:
        small_diff_count += 1
        logger(f"   >>> 微小变化检测: 连续 {small_diff_count} 次 F1 变化 <= {small_diff_threshold}")
    else:
        small_diff_count = 0

    return True, {
        "prompt": new_prompt,
        "y_pred": new_y_pred,
        "small_diff_count": small_diff_count,
        "large_drop_count": large_drop_count
    }


async def run_single_experiment(
        ENTITY_TYPES, forget_num, data, eval_data, client, META_PROMPT,
        max_iterations, max_per_class, patience, max_retries,
        small_diff_threshold, drop_threshold, num_index,
        dataset, model_name, fusion_stability_threshold=0.1, fusion_stability_patience=2, large_drop_patience=3,
        max_sample_count=3, test_f1s=None
):
    """
    单次实验：固定 small_diff_threshold 和 drop_threshold，
    运行完整的迭代流程，返回每轮 fusion 的 macro_f1 列表。
    """
    iteration_f1s = []

    # 全局预测池：记录所有专家在所有轮次的预测
    # {text: [pred_comp_iter0, pred_parse_iter0, pred_exam_iter0, pred_fuse_iter0,
    #         pred_comp_iter1, pred_parse_iter1, ...]}
    global_prediction_pool = {}

    # 保留原有的融合历史（用于其他逻辑）
    prediction_history = {}  # {text: [entities_iter0, entities_iter1, ...]}

    # 新增：采样历史追踪（避免过拟合）- 三个专家分别维护
    sampling_history_comp = {}  # Compare 专家的采样历史
    sampling_history_parse = {}  # Parsing 专家的采样历史
    sampling_history_exam = {}  # Example 专家的采样历史

    # 新增：跟踪每一轮伪标签质量指标
    pseudo_label_metrics_history = []

    # 新增：跟踪每一轮每个专家的指令历史
    prompt_history = {
        "compare_agent": [],
        "parsing_agent": [],
        "example_agent": [],
        "metadata": {
            "dataset": dataset,
            "model_name": model_name,
            "small_diff_threshold": small_diff_threshold,
            "drop_threshold": drop_threshold,
            "num_index": num_index
        }
    }

    print(f"\n{'=' * 60}")
    print(
        f"[Experiment] sdt={small_diff_threshold}, dt={drop_threshold}, run={num_index}")
    print(f"{'=' * 60}")

    # --- 优化1：初始三专家并行预测 ---
    print("   >>> [优化] 初始三专家并行预测...")
    results = await asyncio.gather(
        compare(ENTITY_TYPES, data, [], client, max_retries, META_PROMPT, META_PROMPT, None, True),
        parsing(ENTITY_TYPES, data, [], client, max_retries, META_PROMPT, META_PROMPT, None, True),
        example(ENTITY_TYPES, data, [], client, max_retries, META_PROMPT, META_PROMPT, None, True)
    )
    print(META_PROMPT)
    y_pred_1, _, e1 = results[0]
    y_pred_2, _, e2 = results[1]
    y_pred_3, _, e3 = results[2]
    cur_prompt = META_PROMPT

    # 更新全局预测池（第0轮）- 记录所有专家的预测
    for item in y_pred_1:
        text = item["text"]
        global_prediction_pool.setdefault(text, []).append(item.get("entities", []))
    for item in y_pred_2:
        text = item["text"]
        global_prediction_pool.setdefault(text, []).append(item.get("entities", []))
    for item in y_pred_3:
        text = item["text"]
        global_prediction_pool.setdefault(text, []).append(item.get("entities", []))

    # 保留原有的融合历史更新（用于其他逻辑）
    prediction_history = update_prediction_history(prediction_history, y_pred_1)
    prediction_history = update_prediction_history(prediction_history, y_pred_2)
    prediction_history = update_prediction_history(prediction_history, y_pred_3)

    print(e1)
    print(e2)
    print(e3)

    # 其他部分均是在训练数据上执行，这里电镀在测试数据上执行，观察在测试集上的性能
    print("*************************测试 - Fusion-{}-{}************************".format(0, num_index))
    _m0 = await expert_fusion(
        ENTITY_TYPES, eval_data, client, max_retries,
        META_PROMPT, eval_data,
        META_PROMPT, eval_data,
        META_PROMPT, eval_data,
        dataset, 0, num_index, model_name, small_diff_threshold,
        drop_threshold, if_log=True, stage_train=False)
    _t0 = calculate_metrics_all(ENTITY_TYPES, eval_data, _m0)
    if test_f1s is not None:
        test_f1s.append({"macro_f1": _t0["macro_f1"], "micro_f1": _t0["micro_f1"]})

    # 第 0 轮 fusion：1 次确定性融合（temperature=0，兼作类型平票裁决）+ 2 次 temperature=0.5 融合，多数投票
    init_temp = 0.5
    print(f"   >>> [初始化] 第0轮融合 3 次（temperature=0 x1, {init_temp} x2）...")
    fusion_runs = await asyncio.gather(
        expert_fusion(ENTITY_TYPES, data, client, max_retries, cur_prompt, y_pred_1,
                      cur_prompt, y_pred_2, cur_prompt, y_pred_3, dataset, 0, num_index, model_name,
                      small_diff_threshold, drop_threshold, temp=0.0),
        expert_fusion(ENTITY_TYPES, data, client, max_retries, cur_prompt, y_pred_1,
                      cur_prompt, y_pred_2, cur_prompt, y_pred_3, dataset, 0, num_index, model_name,
                      small_diff_threshold, drop_threshold, temp=init_temp),
        expert_fusion(ENTITY_TYPES, data, client, max_retries, cur_prompt, y_pred_1,
                      cur_prompt, y_pred_2, cur_prompt, y_pred_3, dataset, 0, num_index, model_name,
                      small_diff_threshold, drop_threshold, temp=init_temp),
    )

    # 多数投票（K//2+1）+ 有效性过滤 + 同 span 单类型，作为初始化锚点
    prev_merged = init_anchor_from_fusion_runs(fusion_runs, ENTITY_TYPES)

    # 更新全局预测池 - 添加融合结果
    for item in prev_merged:
        text = item["text"]
        global_prediction_pool.setdefault(text, []).append(item.get("entities", []))

    print(f"   >>> [初始化] 多数投票锚点实体数: {sum(len(item['entities']) for item in prev_merged)}")
    fusion_stable_count = 0

    # 直接使用融合结果作为锚点集合
    data_update = prev_merged

    # 计算初始伪标签质量指标（第0轮）
    initial_pseudo_metrics = evaluate_pseudo_labels(data_update, data, ENTITY_TYPES)
    pseudo_label_metrics_history.append({
        "iteration": 0,
        **initial_pseudo_metrics
    })
    print(f"   >>> [第0轮伪标签指标] Precision: {initial_pseudo_metrics['pseudo_label_precision']:.4f}, "
          f"Recall: {initial_pseudo_metrics['pseudo_label_recall']:.4f}, "
          f"F1: {initial_pseudo_metrics['pseudo_label_f1']:.4f}, "
          f"FP Count: {initial_pseudo_metrics['false_positive_count']}, "
          f"FPR: {initial_pseudo_metrics['false_positive_rate']:.4f}")

    # 第一轮错误采样会使用兜底采样策略，也就是如果这类错误没有，也会采样部分案例迫使其必须按照专家视角改写。
    loop = asyncio.get_event_loop()
    sampled_results = await asyncio.gather(
        loop.run_in_executor(None, stratified_sample_errors, ENTITY_TYPES, data_update, y_pred_1,
                             max_per_class, None, None, True, global_prediction_pool, False,
                             None, sampling_history_comp, max_sample_count),
        loop.run_in_executor(None, stratified_sample_errors, ENTITY_TYPES, data_update, y_pred_2,
                             max_per_class, None, None, True, global_prediction_pool, False,
                             None, sampling_history_parse, max_sample_count),
        loop.run_in_executor(None, stratified_sample_errors, ENTITY_TYPES, data_update, y_pred_3,
                             max_per_class, None, None, True, global_prediction_pool, False,
                             None, sampling_history_exam, max_sample_count)
    )

    # 【修改2】初始采样后更新历史计数
    sampled_errors_comp_w_init = sampled_results[0]["predict_error_entities_w_label"]
    sampled_errors_parse_w_init = sampled_results[1]["recall_error_entities_w_label"]
    sampled_errors_exam_w_init = sampled_results[2]["all_error_entities_w_label_gold"]

    for text in sampled_errors_comp_w_init.keys():
        k = sampling_key(text)
        sampling_history_comp[k] = sampling_history_comp.get(k, 0) + 1
    for text in sampled_errors_parse_w_init.keys():
        k = sampling_key(text)
        sampling_history_parse[k] = sampling_history_parse.get(k, 0) + 1
    for text in sampled_errors_exam_w_init.keys():
        k = sampling_key(text)
        sampling_history_exam[k] = sampling_history_exam.get(k, 0) + 1
    sampled_errors_comp_w = sampled_results[0]["predict_error_entities_w_label"]
    sampled_errors_parse_w = sampled_results[1]["recall_error_entities_w_label"]
    sampled_errors_exam_w = sampled_results[2]["all_error_entities_w_label_gold"]

    # 初始化专家状态（方案A：不缓存metrics，每次在update_expert_with_safeguard内实时计算）
    state_comp = {"prompt": cur_prompt, "y_pred": y_pred_1, "small_diff_count": 0, "large_drop_count": 0}
    state_parse = {"prompt": cur_prompt, "y_pred": y_pred_2, "small_diff_count": 0, "large_drop_count": 0}
    state_exam = {"prompt": cur_prompt, "y_pred": y_pred_3, "small_diff_count": 0, "large_drop_count": 0}

    # 新增：记录第0轮的初始prompt
    prompt_history["compare_agent"].append({
        "iteration": 0,
        "prompt": cur_prompt,
        "is_accepted": True,
        "note": "初始prompt"
    })
    prompt_history["parsing_agent"].append({
        "iteration": 0,
        "prompt": cur_prompt,
        "is_accepted": True,
        "note": "初始prompt"
    })
    prompt_history["example_agent"].append({
        "iteration": 0,
        "prompt": cur_prompt,
        "is_accepted": True,
        "note": "初始prompt"
    })

    overfit_warn_count = 0  # 连续出现过拟合提示的轮数
    for iteration in range(1, max_iterations):
        predict_list = []

        print(f"   >>> [优化] 第{iteration}轮三专家并行更新...")
        tasks = []
        task_info = []

        # Compare-Agent
        if state_comp["small_diff_count"] < patience and state_comp["large_drop_count"] < large_drop_patience:
            if not sampled_errors_comp_w:
                print("   >>> Compare-Agent: errors为空，跳过更新（不加入predict_list）")
                _, new_state = update_expert_with_safeguard(
                    "Compare-Agent", state_comp, state_comp["prompt"], state_comp["y_pred"],
                    data_update, ENTITY_TYPES, iteration, 3, small_diff_threshold, drop_threshold, large_drop_patience)
                state_comp = new_state
            else:
                tasks.append(compare(ENTITY_TYPES, data, data_update, client, max_retries,
                                     state_comp["prompt"], state_comp["prompt"], sampled_errors_comp_w, False,
                                     state_comp["y_pred"],
                                     prediction_history, global_prediction_pool))
                task_info.append(("compare", state_comp))

        # Parsing-Agent
        if state_parse["small_diff_count"] < patience and state_parse["large_drop_count"] < large_drop_patience:
            if not sampled_errors_parse_w:
                print("   >>> Parsing-Agent: errors为空，跳过更新（不加入predict_list）")
                _, new_state = update_expert_with_safeguard(
                    "Parsing-Agent", state_parse, state_parse["prompt"], state_parse["y_pred"],
                    data_update, ENTITY_TYPES, iteration, 3, small_diff_threshold, drop_threshold, large_drop_patience)
                state_parse = new_state
            else:
                tasks.append(parsing(ENTITY_TYPES, data, data_update, client, max_retries,
                                     state_parse["prompt"], state_parse["prompt"], sampled_errors_parse_w, False,
                                     state_parse["y_pred"],
                                     prediction_history, global_prediction_pool))
                task_info.append(("parsing", state_parse))

        # Example-Agent
        if state_exam["small_diff_count"] < patience and state_exam["large_drop_count"] < large_drop_patience:
            if not sampled_errors_exam_w:
                print("   >>> Example-Agent: errors为空，跳过更新（不加入predict_list）")
                _, new_state = update_expert_with_safeguard(
                    "Example-Agent", state_exam, state_exam["prompt"], state_exam["y_pred"],
                    data_update, ENTITY_TYPES, iteration, 3, small_diff_threshold, drop_threshold, large_drop_patience)
                state_exam = new_state
            else:
                tasks.append(example(ENTITY_TYPES, data, data_update, client, max_retries,
                                     state_exam["prompt"], state_exam["prompt"], sampled_errors_exam_w, False,
                                     state_exam["y_pred"],
                                     prediction_history, global_prediction_pool))
                task_info.append(("example", state_exam))

        # 并行执行有errors的专家
        if tasks:
            parallel_results = await asyncio.gather(*tasks)
            for (expert_type, state), (y_pred_new, proposed_prompt, _) in zip(task_info, parallel_results):
                is_accepted, new_state = update_expert_with_safeguard(
                    f"{expert_type.capitalize()}-Agent", state, proposed_prompt, y_pred_new,
                    data_update, ENTITY_TYPES, iteration, 3, small_diff_threshold, drop_threshold, large_drop_patience)

                # 新增：记录专家的prompt更新d
                if expert_type == "compare":
                    state_comp = new_state
                    prompt_history["compare_agent"].append({
                        "iteration": iteration,
                        "prompt": new_state["prompt"],
                        "is_accepted": is_accepted,
                        "small_diff_count": new_state["small_diff_count"],
                        "large_drop_count": new_state["large_drop_count"]
                    })
                elif expert_type == "parsing":
                    state_parse = new_state
                    prompt_history["parsing_agent"].append({
                        "iteration": iteration,
                        "prompt": new_state["prompt"],
                        "is_accepted": is_accepted,
                        "small_diff_count": new_state["small_diff_count"],
                        "large_drop_count": new_state["large_drop_count"]
                    })
                else:
                    state_exam = new_state
                    prompt_history["example_agent"].append({
                        "iteration": iteration,
                        "prompt": new_state["prompt"],
                        "is_accepted": is_accepted,
                        "small_diff_count": new_state["small_diff_count"],
                        "large_drop_count": new_state["large_drop_count"]
                    })
                predict_list.append(new_state["y_pred"])

        # 更新全局预测池（本轮迭代）- 添加所有专家的预测
        if predict_list:
            for pred in [state_comp["y_pred"], state_parse["y_pred"], state_exam["y_pred"]]:
                for item in pred:
                    text = item["text"]
                    global_prediction_pool.setdefault(text, []).append(item.get("entities", []))

        # 保留原有的融合历史更新（用于其他逻辑）
        if predict_list:
            for pred in [state_comp["y_pred"], state_parse["y_pred"], state_exam["y_pred"]]:
                prediction_history = update_prediction_history(prediction_history, pred)

        # 所有专家都已停止（收敛 or 大幅波动超限），终止整体迭代
        all_stopped = (
                (state_comp["small_diff_count"] >= patience or state_comp[
                    "large_drop_count"] >= large_drop_patience) and
                (state_parse["small_diff_count"] >= patience or state_parse[
                    "large_drop_count"] >= large_drop_patience) and
                (state_exam["small_diff_count"] >= patience or state_exam["large_drop_count"] >= large_drop_patience)
        )

        #############################测试################################
        # 其他部分均是在训练数据上执行，这里电镀在测试数据上执行，观察在测试集上的性能
        print("*************************测试 - Fusion-{}-{}************************".format(iteration, num_index))
        _mt = await expert_fusion(
            ENTITY_TYPES, eval_data, client, max_retries,
            state_comp["prompt"], eval_data,
            state_parse["prompt"], eval_data,
            state_exam["prompt"], eval_data,
            dataset, iteration, num_index, model_name, small_diff_threshold,
            drop_threshold, if_log=True, stage_train=False)
        _tm = calculate_metrics_all(ENTITY_TYPES, eval_data, _mt)
        if test_f1s is not None:
            test_f1s.append({"macro_f1": _tm["macro_f1"], "micro_f1": _tm["micro_f1"]})

        if iteration == forget_num:
            state_comp["small_diff_count"] = 0
            state_parse["small_diff_count"] = 0
            state_exam["small_diff_count"] = 0
            state_comp["large_drop_count"] = 0
            state_parse["large_drop_count"] = 0
            state_exam["large_drop_count"] = 0
            state_comp["prompt"] = META_PROMPT
            state_parse["prompt"] = META_PROMPT
            state_exam["prompt"] = META_PROMPT
        ##############################################################

        if all_stopped:
            pseudo_metrics = evaluate_pseudo_labels(data_update, data, ENTITY_TYPES)
            print(f"   >>> [第{iteration}轮伪标签指标] Precision: {pseudo_metrics['pseudo_label_precision']:.4f}, "
                  f"Recall: {pseudo_metrics['pseudo_label_recall']:.4f}, "
                  f"F1: {pseudo_metrics['pseudo_label_f1']:.4f}, "
                  f"FP Count: {pseudo_metrics['false_positive_count']}, "
                  f"FPR: {pseudo_metrics['false_positive_rate']:.4f}, "
                  f"Error Rate: {pseudo_metrics['pseudo_label_error_rate']:.4f}")
            print("所有专家均已停止迭代（收敛或大幅波动），退出。")
            break
        # Fusion & 记录本轮 F1
        # print("*************************训练 - Fusion-{}-{}************************".format(iteration, num_index))
        if iteration != forget_num:

            merged = await expert_fusion(
                ENTITY_TYPES, data, client, max_retries,
                state_comp["prompt"], state_comp["y_pred"],
                state_parse["prompt"], state_parse["y_pred"],
                state_exam["prompt"], state_exam["y_pred"],
                dataset, iteration, num_index, model_name, small_diff_threshold,
                drop_threshold)

            # 更新全局预测池 - 添加融合结果
            for item in merged:
                text = item["text"]
                global_prediction_pool.setdefault(text, []).append(item.get("entities", []))

            fusion_metrics = calculate_metrics_all(ENTITY_TYPES, data, merged)
            iteration_f1s.append({
                "iteration": iteration,
                "macro_f1": fusion_metrics["macro_f1"],
                "macro_precision": fusion_metrics["macro_precision"],
                "macro_recall": fusion_metrics["macro_recall"],
                "micro_f1": fusion_metrics["micro_f1"],
                "micro_precision": fusion_metrics["micro_precision"],
                "micro_recall": fusion_metrics["micro_recall"]
            })

            # 融合结果整体稳定性检测
            fusion_change = compute_fusion_change(prev_merged, merged)
            print(f"   >>> 融合结果变化率: {fusion_change:.4f} (阈值: {fusion_stability_threshold})")
            if fusion_change < fusion_stability_threshold:
                fusion_stable_count += 1
                print(f"   >>> 融合结果连续稳定 {fusion_stable_count}/{fusion_stability_patience} 次")
            else:
                fusion_stable_count = 0
            prev_merged = merged

            if fusion_stable_count >= fusion_stability_patience:
                print(
                    f"融合结果连续 {fusion_stability_patience} 次变化率低于 {fusion_stability_threshold}，整体收敛，提前终止。")
                break

        if predict_list:
            # 策略3：保守锚点更新，只更新高共识实体
            if iteration == forget_num:
                data_update = data_update
            else:
                data_update = conservative_anchor_update(
                    data_update, merged,
                    [state_comp["y_pred"], state_parse["y_pred"], state_exam["y_pred"]],
                    consensus_threshold=2
                )
                # 计算保守更新与直接替换的差异
                conservative_change = compute_fusion_change(data_update, merged)
                print(f"   >>> 保守锚点更新: 与直接替换的差异率={conservative_change:.4f}")

                # 计算并记录本轮伪标签质量指标
                pseudo_metrics = evaluate_pseudo_labels(data_update, data, ENTITY_TYPES)
                pseudo_label_metrics_history.append({
                    "iteration": iteration,
                    **pseudo_metrics
                })
                print(f"   >>> [第{iteration}轮伪标签指标] Precision: {pseudo_metrics['pseudo_label_precision']:.4f}, "
                      f"Recall: {pseudo_metrics['pseudo_label_recall']:.4f}, "
                      f"F1: {pseudo_metrics['pseudo_label_f1']:.4f}, "
                      f"FP Count: {pseudo_metrics['false_positive_count']}, "
                      f"FPR: {pseudo_metrics['false_positive_rate']:.4f}, "
                      f"Error Rate: {pseudo_metrics['pseudo_label_error_rate']:.4f}")
        else:
            print("   >>> predict_list 为空，跳过 anchor 集更新，保留上轮 anchor 集")

        sampled_results = await asyncio.gather(
            loop.run_in_executor(None, stratified_sample_errors, ENTITY_TYPES, data_update,
                                 state_comp["y_pred"], max_per_class, None, None,
                                 False, global_prediction_pool, iteration >= 2, iteration + 1,
                                 sampling_history_comp, max_sample_count),
            loop.run_in_executor(None, stratified_sample_errors, ENTITY_TYPES, data_update,
                                 state_parse["y_pred"], max_per_class, None, None,
                                 False, global_prediction_pool, iteration >= 2, iteration + 1,
                                 sampling_history_parse, max_sample_count),
            loop.run_in_executor(None, stratified_sample_errors, ENTITY_TYPES, data_update,
                                 state_exam["y_pred"], max_per_class, None, None, False,
                                 global_prediction_pool, iteration >= 2, iteration + 1,
                                 sampling_history_exam, max_sample_count)
        )
        sampled_errors_comp_w = sampled_results[0]["predict_error_entities_w_label"]
        sampled_errors_parse_w = sampled_results[1]["recall_error_entities_w_label"]
        sampled_errors_exam_w = sampled_results[2]["all_error_entities_w_label_gold"]

        # 更新采样历史 - 三个专家分别维护
        for text in sampled_errors_comp_w.keys():
            k = sampling_key(text)
            sampling_history_comp[k] = sampling_history_comp.get(k, 0) + 1
        for text in sampled_errors_parse_w.keys():
            k = sampling_key(text)
            sampling_history_parse[k] = sampling_history_parse.get(k, 0) + 1

        # all_error_entities_w_label_gold 是 {原句: [错误元组, ...]}
        for text in sampled_errors_exam_w.keys():
            k = sampling_key(text)
            sampling_history_exam[k] = sampling_history_exam.get(k, 0) + 1

        # 【修改1】按专家分别统计重复率
        total_comp = sum(sampling_history_comp.values())
        total_parse = sum(sampling_history_parse.values())
        total_exam = sum(sampling_history_exam.values())
        unique_comp = len(sampling_history_comp)
        unique_parse = len(sampling_history_parse)
        unique_exam = len(sampling_history_exam)

        repeat_rate_comp = 1 - unique_comp / total_comp if total_comp > 0 else 0
        repeat_rate_parse = 1 - unique_parse / total_parse if total_parse > 0 else 0
        repeat_rate_exam = 1 - unique_exam / total_exam if total_exam > 0 else 0

        if total_comp > 0 or total_parse > 0 or total_exam > 0:
            print(f"[第{iteration}轮采样监控]")
            print(f"   Compare专家: {total_comp}次/{unique_comp}样本, 重复率:{repeat_rate_comp:.1%}")
            print(f"   Parsing专家: {total_parse}次/{unique_parse}样本, 重复率:{repeat_rate_parse:.1%}")
            print(f"   Example专家: {total_exam}次/{unique_exam}样本, 重复率:{repeat_rate_exam:.1%}")

            # 计算平均重复率用于过拟合预警
            avg_repeat_rate = (repeat_rate_comp + repeat_rate_parse + repeat_rate_exam) / 3
            if avg_repeat_rate > 0.50:
                print(f"   ⚠️ 注意：平均重复率较高({avg_repeat_rate:.1%})，可能过拟合！")
            elif avg_repeat_rate > 0.30:
                print(f"   ℹ️ 提示：平均重复率中等({avg_repeat_rate:.1%})，需关注")

            # 终止条件：连续两轮出现过拟合提示（平均重复率 > 0.50）则停止迭代
            # overfit_warn_count = overfit_warn_count + 1 if avg_repeat_rate > 0.50 else 0
            # if overfit_warn_count >= 2:
            #     print(f"   >>> 连续 {overfit_warn_count} 轮出现过拟合提示，终止迭代。")
            #     break

            # Top 5 最常采样的样例（合并统计）
            combined_history = {}
            for hist in [sampling_history_comp, sampling_history_parse, sampling_history_exam]:
                for text, count in hist.items():
                    combined_history[text] = combined_history.get(text, 0) + count

    # 新增：保存prompt历史到文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prompt_history_dir = f"./prompt_history/{model_name}/{dataset}/sdt{small_diff_threshold}_dt{drop_threshold}"
    os.makedirs(prompt_history_dir, exist_ok=True)
    prompt_history_path = f"{prompt_history_dir}/{timestamp}_iter{iteration}_run{num_index}_prompts.json"

    with open(prompt_history_path, "w", encoding="utf-8") as f:
        json.dump(prompt_history, f, ensure_ascii=False, indent=4)
    print(f"\n>>> Prompt历史已保存至: {prompt_history_path}")

    # 新增：保存伪标签指标历史到文件
    pseudo_metrics_dir = f"./pseudo_label_metrics/{model_name}/{dataset}/sdt{small_diff_threshold}_dt{drop_threshold}"
    os.makedirs(pseudo_metrics_dir, exist_ok=True)
    pseudo_metrics_path = f"{pseudo_metrics_dir}/{timestamp}_iter{iteration}_run{num_index}_pseudo_metrics.json"

    with open(pseudo_metrics_path, "w", encoding="utf-8") as f:
        json.dump(pseudo_label_metrics_history, f, ensure_ascii=False, indent=4)
    print(f">>> 伪标签指标历史已保存至: {pseudo_metrics_path}")

    return iteration_f1s


async def main(model_name, datasets, base_url, small_diff_thresholds, drop_thresholds, num_runs, patience,
               max_concurrent=60, max_sample_count=3):
    # 固定超参
    max_iterations = 10
    max_per_class = 3  # 【修改3】从5提升到12，让每轮专家能够采样更多样本充分学习
    patience = patience
    max_retries = 3
    forget_num = 0

    global MODELNAME
    global ENTITY_TYPES
    MODELNAME = model_name

    # 初始化全局API并发上限
    set_global_api_concurrency(max_concurrent)
    print(f"[GlobalSem] 全局API并发上限设置为 {max_concurrent}")

    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=base_url
    )

    for dataset in datasets:
        print(f"\n{'=' * 50}")
        print(f"开始处理数据集: {dataset}")
        print(f"{'=' * 50}\n")

        # 根据数据集设置对应的配置
        if dataset == "GENIA":
            ENTITY_TYPES = ENTITY_TYPE_genia.keys()
            ENTITY_TYPES_d = ENTITY_TYPE_genia
            META_PROMPT = PROMPT_TEMPLATE_genia

        # 构建数据文件路径（相对于脚本所在目录的父目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        test_data_path = os.path.join(parent_dir, "datas", dataset, "test_200.json")
        train_data_path = os.path.join(parent_dir, "datas", dataset, "train.json")

        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
            test_data = convert_label(test_data, dataset)

        with open(train_data_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
            train_data = convert_label(train_data, dataset)

        # 汇总结果：{(ws, vt, sdt, dt): {"runs": [...], "best_f1": float, "avg_f1": float}}
        search_results = {}

        # 计算有效参数组合数量（考虑约束：vote_threshold <= window_size）
        valid_combinations = [
            (sdt, dt)
            for sdt in small_diff_thresholds
            for dt in drop_thresholds
            # if vt <= ws  # 约束条件：投票阈值不能超过窗口大小
        ]
        print(valid_combinations)
        total = len(valid_combinations)
        done = 0

        for small_diff_threshold, drop_threshold in valid_combinations:
            done += 1
            print(f"\n[{done}/{total}] small_diff_threshold={small_diff_threshold}, drop_threshold={drop_threshold}")
            key = (small_diff_threshold, drop_threshold)
            search_results[key] = {
                "runs": [],
                "best_macro_f1": 0.0,
                "avg_macro_f1": 0.0,
                "best_micro_f1": 0.0,
                "avg_micro_f1": 0.0
            }

            for num_index in range(num_runs):
                num_index += 1
                eval_subset = test_data
                train_subset = train_data
                test_f1s = []
                iteration_f1s = await run_single_experiment(
                    ENTITY_TYPES, forget_num, train_subset, eval_subset, client,
                    META_PROMPT,
                    max_iterations, max_per_class, patience, max_retries,
                    small_diff_threshold, drop_threshold, num_index,
                    dataset, MODELNAME, max_sample_count=max_sample_count, test_f1s=test_f1s
                )
                search_results[key]["runs"].append(iteration_f1s)
                # 取本次实验最高 macro_f1 和 micro_f1
                run_best_macro = max((r["macro_f1"] for r in iteration_f1s), default=0.0)
                run_best_micro = max((r["micro_f1"] for r in iteration_f1s), default=0.0)
                if run_best_macro > search_results[key]["best_macro_f1"]:
                    search_results[key]["best_macro_f1"] = run_best_macro
                if run_best_micro > search_results[key]["best_micro_f1"]:
                    search_results[key]["best_micro_f1"] = run_best_micro

                macro_traj = " -> ".join(f"{r['macro_f1']:.4f}" for r in test_f1s)
                micro_traj = " -> ".join(f"{r['micro_f1']:.4f}" for r in test_f1s)
                print(f"  [run {num_index}] 测试Fusion macro F1 轨迹：{macro_traj}")
                print(f"  [run {num_index}] 测试Fusion micro F1 轨迹：{micro_traj}")
                print(f"  [run {num_index}] best_macro_f1={run_best_macro:.4f}, best_micro_f1={run_best_micro:.4f}")

            # 计算多次运行的平均最高 F1
            run_bests_macro = [max((r["macro_f1"] for r in run), default=0.0)
                               for run in search_results[key]["runs"]]
            run_bests_micro = [max((r["micro_f1"] for r in run), default=0.0)
                               for run in search_results[key]["runs"]]
            search_results[key]["avg_macro_f1"] = sum(run_bests_macro) / len(
                run_bests_macro) if run_bests_macro else 0.0
            search_results[key]["avg_micro_f1"] = sum(run_bests_micro) / len(
                run_bests_micro) if run_bests_micro else 0.0

            print(f"  [sdt={small_diff_threshold}, dt={drop_threshold}] "
                  f"avg_macro_f1={search_results[key]['avg_macro_f1']:.4f}, avg_micro_f1={search_results[key]['avg_micro_f1']:.4f}")

        # 打印最优参数组合（基于 macro_f1）
        best_key_macro = max(search_results, key=lambda k: search_results[k]["avg_macro_f1"])
        print(f"\n最优参数 (Macro-F1): small_diff_threshold={best_key_macro[0]}, "
              f"drop_threshold={best_key_macro[1]}, "
              f"avg_macro_f1={search_results[best_key_macro]['avg_macro_f1']:.4f}, "
              f"avg_micro_f1={search_results[best_key_macro]['avg_micro_f1']:.4f}")

        # 打印最优参数组合（基于 micro_f1）
        best_key_micro = max(search_results, key=lambda k: search_results[k]["avg_micro_f1"])
        print(f"\n最优参数 (Micro-F1): small_diff_threshold={best_key_micro[0]}, "
              f"drop_threshold={best_key_micro[1]}, "
              f"avg_macro_f1={search_results[best_key_micro]['avg_macro_f1']:.4f}, "
              f"avg_micro_f1={search_results[best_key_micro]['avg_micro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行NER分析实验')
    # parser.add_argument('--model', type=str, default="gpt-3.5-turbo-0125", help='模型名称')
    parser.add_argument('--model', type=str, default="Qwen32b", help='模型名称')
    parser.add_argument('--datasets', type=str, default="GENIA", help='数据集名称（逗号分隔）')
    parser.add_argument('--base_url', type=str, default="http://localhost:8000/v1", help='API base URL')
    parser.add_argument('--small_diff_thresholds', type=str, default="0.04",
                        help='small_diff阈值列表（逗号分隔）')
    parser.add_argument('--drop_thresholds', type=str, default="0.4",
                        help='drop阈值列表（逗号分隔）')
    parser.add_argument('--num_runs', type=int, default=6, help='每组参数重复次数')
    parser.add_argument('--patience', type=int, default=3, help='每组参数重复次数')
    parser.add_argument('--max_concurrent', type=int, default=100,
                        help='全局API并发上限（覆盖所有 entity_pred/summarize/fuse_entities）')
    parser.add_argument('--example_num', type=int, default=EXAMPLE_NUM,
                        help='优化后指令保留的代表性样例数（聚类数），<=0 关闭精简')
    parser.add_argument('--max_sample_count', type=int, default=3,
                        help='单个样本最多采样次数（硬性上限），默认3次')

    args = parser.parse_args()
    EXAMPLE_NUM = args.example_num

    # 解析参数
    datasets = [d.strip() for d in args.datasets.split(',')]
    small_diff_thresholds = [float(x.strip()) for x in args.small_diff_thresholds.split(',')]
    drop_thresholds = [float(x.strip()) for x in args.drop_thresholds.split(',')]

    asyncio.run(main(
        model_name=args.model,
        datasets=datasets,
        base_url=args.base_url,
        small_diff_thresholds=small_diff_thresholds,
        drop_thresholds=drop_thresholds,
        num_runs=args.num_runs,
        patience=args.patience,
        max_concurrent=args.max_concurrent,
        max_sample_count=args.max_sample_count
    ))
