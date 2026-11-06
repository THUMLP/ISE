import random
from collections import defaultdict, Counter
import json

import re

# stop_words = set(stopwords.words('english'))


def normalize_entities_output(entities_data):
    """
    统一解析模型实体输出，兼容以下格式：
    1) {"type1": ["v1", "v2"], "type2": ["v3"]}
    2) [{"type": "xxx", "entity": "yyy"}, ...]
    3) [{"type1": "v1"}, {"type2": "v2"}, ...]
    4) [{"type1": ["v1", "v2"]}, ...]
    返回统一内部格式：[{"type": "...", "entity": "..."}, ...]
    """
    normalized = []

    def _append_entity(ent_type, ent_value):
        if ent_type is None or ent_value is None:
            return
        ent_type = str(ent_type).strip()
        ent_value = str(ent_value).strip()
        if ent_type and ent_value:
            normalized.append({"type": ent_type, "entity": ent_value})

    if isinstance(entities_data, dict):
        for ent_type, ent_values in entities_data.items():
            if isinstance(ent_values, list):
                for value in ent_values:
                    _append_entity(ent_type, value)
            else:
                _append_entity(ent_type, ent_values)
        return normalized

    if isinstance(entities_data, list):
        for ent in entities_data:
            if not isinstance(ent, dict):
                continue

            if "type" in ent and "entity" in ent:
                _append_entity(ent.get("type"), ent.get("entity"))
                continue

            if len(ent) == 1:
                ent_type, ent_value = next(iter(ent.items()))
                if isinstance(ent_value, list):
                    for value in ent_value:
                        _append_entity(ent_type, value)
                else:
                    _append_entity(ent_type, ent_value)

    return normalized


def calculate_metrics_all(ENTITY_TYPES, y_true, y_pred):
    # 1. 初始化每个实体类型的统计计数器
    # 格式: {'address': {'TP': 0, 'FP': 0, 'FN': 0}, 'name': ...}
    target_types = set(t.lower() for t in ENTITY_TYPES)
    type_stats = {t: {'TP': 0, 'FP': 0, 'FN': 0} for t in target_types}

    # 将 y_pred 转换为字典，以文本内容为键，方便快速查找
    pred_dict = {item["text"]: item for item in y_pred}

    for true_item in y_true:
        text = true_item["text"]

        # 获取当前文本的真实实体集合 {(type, entity), ...}
        # 仅保留在 ENTITY_TYPES 中的类型
        true_entities = set()
        for ent in true_item["entities"]:
            t_type = ent["type"].lower()
            if t_type in target_types:
                # 实体文本也转为小写，不区分大小写
                entity_text = ent["entity"].lower()
                true_entities.add((t_type, entity_text))

        # 获取当前文本的预测实体集合
        pred_entities = set()
        if text in pred_dict:
            for ent in pred_dict[text]["entities"]:
                # 增加数据格式校验，防止报错
                if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                    t_type = ent["type"].lower()
                    if t_type in target_types:
                        try:
                            # 实体文本也转为小写，不区分大小写
                            entity_text = ent["entity"].lower()
                            pred_entities.add((t_type, entity_text))
                        except:
                            pass

        # 2. 计算当前样本中，每个类别的 TP, FP, FN

        # TP: 预测正确 (交集)
        # 遍历交集，根据具体的 type 增加对应的 TP 计数
        for t, _ in true_entities & pred_entities:
            type_stats[t]['TP'] += 1

        # FP: 预测了但不在真实集中 (预测集 - 真实集)
        for t, _ in pred_entities - true_entities:
            type_stats[t]['FP'] += 1

        # FN: 真实存在但未被预测 (真实集 - 预测集)
        # 注意：如果 text 不在 pred_dict 中，pred_entities 为空，这里会自动处理为全部 FN
        for t, _ in true_entities - pred_entities:
            type_stats[t]['FN'] += 1

    # 3. 计算每个类型的 P, R, F1，并收集起来
    p_list, r_list, f1_list = [], [], []

    for t in target_types:
        tp = type_stats[t]['TP']
        fp = type_stats[t]['FP']
        fn = type_stats[t]['FN']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        p_list.append(precision)
        r_list.append(recall)
        f1_list.append(f1)

    # 4. 计算 Macro Average (对所有类别的指标求算术平均)
    macro_precision = sum(p_list) / len(p_list) if p_list else 0
    macro_recall = sum(r_list) / len(r_list) if r_list else 0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0

    # 5. 计算 Micro Average (对所有类别的 TP/FP/FN 求和后计算)
    total_tp = sum(type_stats[t]['TP'] for t in target_types)
    total_fp = sum(type_stats[t]['FP'] for t in target_types)
    total_fn = sum(type_stats[t]['FN'] for t in target_types)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (
                                                                                                      micro_precision + micro_recall) > 0 else 0

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "details": type_stats
    }


def text_segment(text, entity):
    # import jieba

    # 尝试找到实体所在的位置
    start_idx = text.find(entity)
    if start_idx == -1:
        return text  # 如果实体未找到，返回原文本或其他适当值

    end_idx = start_idx + len(entity)

    # 判断是否属于中文文本
    def is_chinese(text):
        for char in text:
            if '一' <= char <= '鿿':  # 中文字符范围
                return True
        return False

    # 如果是中文文本，使用 jieba 分词；否则使用 split 分词
    if is_chinese(text):
        # 对实体前后的文本进行分词处理
        # before_entity = list(jieba.cut(text[:start_idx]))  # 实体前分词
        # after_entity = list(jieba.cut(text[end_idx:]))  # 实体后分词

        # 拼接中文文本（无空格）
        # out_str = "..." + "".join(before_entity[-10:]) + entity + "".join(after_entity[:10]) + "..."
        out_str = text
    else:
        # 英文文本直接用 split
        before_entity = text[:start_idx].split()
        after_entity = text[end_idx:].split()

        # 拼接英文文本（用空格隔开）
        # out_str = "..." + " ".join(before_entity[-10:]) + " " + entity + " " + " ".join(after_entity[:10]) + "..."
        out_str = text

    return out_str


def pred_errors_all(ENTITY_TYPES, y_true, y_pred):
    pred_dict = {item["text"]: item for item in y_pred}

    type_errors = []  # 类型错误（标签错误）
    predict_error_by_text = defaultdict(set)
    for true_item in y_true:
        all_errors = []  # 所有错误
        text = true_item["text"]
        try:
            # 保持entity原样，区分大小写
            true_entities = [(ent["type"].lower(), ent["entity"]) for ent in true_item.get("entities", []) if
                             ent.get("type", "").lower() in ENTITY_TYPES]
        except:
            print(true_item)
            break

        if text not in pred_dict:
            continue

        # 获取当前文本下的预测实体列表 [(type, entity_content), ...]
        # 保持entity原样，区分大小写
        pred_entities = [
            (ent["type"].lower(), ent["entity"]) for ent in pred_dict[text]["entities"]
            if isinstance(ent, dict) and "type" in ent and "entity" in ent and ent["type"].lower() in ENTITY_TYPES
        ]
        # ------------------ 补全代码开始 ------------------
        # 遍历该文本下的所有预测实体，验证其准确性
        for p_type, p_content in pred_entities:
            matched_gold_type = None
            is_content_match = False

            # 在真实实体中寻找内容匹配项
            for t_type, t_content in true_entities:
                # 使用题目要求的 fuzzy_match 计算相似度是否达标
                if t_content == p_content:
                    matched_gold_type = t_type
                    is_content_match = True
                    # 找到一个最佳匹配后跳出（贪婪匹配策略），防止重复比对
                    break

            error_record = None

            if is_content_match:
                # 情形1：实体内容匹配上了
                if p_type != matched_gold_type:
                    if p_content in text:
                        # 如果预测类型 != 真实类型 -> (1) 标签错误
                        # 格式: (Predict Type, Entity Content, golden label)
                        sub_text = text_segment(text, p_content)

                        error_record = ["content: " + sub_text, "predict label: {}".format(p_type),
                                        "entity: {}".format(p_content),
                                        "golden label: {}".format(matched_gold_type)]  #
                        type_errors.append(error_record)
            else:
                # 情形2：实体内容没有在真实标签中找到 -> (2) 实体内容不正确
                # 格式: (Predict Type, Entity Content, None)
                if p_content in text:
                    sub_text = text_segment(text, p_content)
                    #
                    # error_record = ["content: " + sub_text, "predict type: {}".format(p_type), "entity: {}".format(p_content),
                    #                 "golden label: Unknown"]  #
                    # entity_errors.append(error_record)

            # 如果产生了错误记录，添加到总错误列表中
            if error_record:
                all_errors.append(error_record)
        # ------------------ 补全代码结束 ------------------
        if all_errors:

            for error in all_errors:
                sub_text = error[0]
                predict_error_by_text[sub_text].add(tuple(error[1:]))

    return {
        "predict_error_by_text": predict_error_by_text
    }


# 分类别统计未召回的实体
def recall_errors_all(ENTITY_TYPES, y_true, y_pred):
    # 建立预测结果字典
    pred_dict = {item["text"]: item for item in y_pred}

    # 结果存储
    recall_error_text = defaultdict(list)  # 建议改为 list 方便存重复错误，如果你确实需要去重可以用 set

    for true_item in y_true:
        text = true_item["text"]

        # 1.获取真实实体列表
        true_entities = []
        for ent in true_item.get("entities", []):
            if "type" in ent and ent["type"].lower() in ENTITY_TYPES and "entity" in ent:
                # 保持entity原样，区分大小写
                true_entities.append((ent["type"].lower(), ent["entity"]))

        # 2.获取预测实体列表 (FIX: 如果 text 不在预测中，则预测实体为空列表，而不是 continue 跳过)
        pred_item = pred_dict.get(text, {})
        pred_entities_pool = []
        if "entities" in pred_item:
            for ent in pred_item["entities"]:
                if isinstance(ent, dict) and "type" in ent and "entity" in ent and ent["type"].lower() in ENTITY_TYPES:
                    # 保持entity原样，区分大小写
                    pred_entities_pool.append((ent["type"].lower(), ent["entity"]))

        # 3.对比逻辑 (FIX: 使用列表移除法，支持同名实体多次出现)
        # 我们使用一个副本或临时列表来模拟"消耗"预测实体的过程
        current_pred_pool = list(pred_entities_pool)

        for true_type, true_val in true_entities:
            # 尝试在预测池中找到一个完全匹配的
            match_found = False
            for i, (pred_type, pred_val) in enumerate(current_pred_pool):
                if pred_type == true_type and pred_val == true_val:
                    # 找到了！从池子中移除它，避免被下一个相同的真实实体重复匹配
                    current_pred_pool.pop(i)
                    match_found = True
                    break

            # 如果遍历完预测池都没找到匹配的，说明丢失了
            if not match_found:
                # 这里的 text_segment 是你的自定义函数，保持原样
                if true_val in text:
                    sub_text = text_segment(text, true_val)

                    # 记录错误
                    # 如果你坚持用 set，请确保 sub_text 或 tuple 足够唯一，可以区分同一句里的两个相同错误
                    recall_error_text["content: " + sub_text].append(
                        ("golden label: {}".format(true_type), "entity: {}".format(true_val), "Easily missed entity"))

    return {"recall_error_text": recall_error_text}


def correct_all(ENTITY_TYPES, y_true, y_pred):
    pred_dict = {item["text"]: item for item in y_pred}
    correct_by_text = defaultdict(set)
    for true_item in y_true:
        text = true_item["text"]
        if text not in pred_dict:
            continue
        true_entities = []
        for ent in true_item["entities"]:
            if "type" in ent and ent["type"].lower() in ENTITY_TYPES and "entity" in ent:
                # 保持entity原样，区分大小写
                true_entities.append((ent["type"].lower(), ent["entity"]))

        # 保持entity原样，区分大小写
        pred_entities = [
            (ent["type"].lower(), ent["entity"]) for ent in pred_dict[text]["entities"]
            if isinstance(ent, dict) and "type" in ent and "entity" in ent and ent["type"].lower() in ENTITY_TYPES
        ]

        for true_type, true_entity in true_entities:
            if any(
                    pred_type == true_type and true_entity == pred_entity
                    for pred_type, pred_entity in pred_entities
            ):
                sub_text = text_segment(text, true_entity)
                correct_by_text["content: " + sub_text].add(
                    ("golden label: {}".format(true_type), true_entity, "Correctly predicted entity"))

    return {"correct_by_text": correct_by_text}


def all_errors(predict_error, recall_error):
    all_error_text = defaultdict(set)
    predict_error_by_text = predict_error["predict_error_by_text"]
    recall_error_text = recall_error["recall_error_text"]

    for text, errors in predict_error_by_text.items():
        for error in errors:
            all_error_text[text].add(error)

    for text, errors in recall_error_text.items():
        for error in errors:
            all_error_text[text].add(error)

    return {"all_error_text": all_error_text}


def convert_to_text_errors(datas):
    text_errors = {}
    for data in datas:
        text, error = data
        if text not in text_errors:
            text_errors[text] = []
        text_errors[text].append(error)
    return text_errors


# 将错误片段归并到所属原句（不做采样；完整标注与错误上下文由
# get_full_entity_examples_from_errors 负责生成）
def get_haserror_candidates(y_true, all_error_entities_w_label, exclude_texts=None):
    """
    返回 [{"text": 原句, "errors": [错误元组, ...]}, ...]
    同一原句对应的多个文本片段的错误会合并并去重。
    """
    y_true_texts = [item["text"] for item in y_true]
    y_true_set = set(y_true_texts)

    merged = {}  # original_text -> candidate

    for text_segment_key, errors in all_error_entities_w_label.items():
        # 去掉前缀 "content: "
        clean_text = text_segment_key.replace("content: ", "")

        # text_segment 可能是句子的一部分或者整句，需要在 y_true 中找到包含它的原句
        if clean_text in y_true_set:
            original_text = clean_text
        else:
            original_text = next((t for t in y_true_texts if clean_text in t), None)

        if original_text is None:
            continue
        if exclude_texts and original_text in exclude_texts:
            continue

        slot = merged.setdefault(original_text, {"text": original_text, "errors": []})
        for err in errors:
            if err not in slot["errors"]:
                slot["errors"].append(err)

    return list(merged.values())

def _norm_label(lb):
    """统一桶键：去掉 'golden label:' 前缀并小写，便于与锚点类型对齐。"""
    return str(lb).replace("golden label:", "").strip().lower()


def filter_by_count_with_reset(items, key_fn, sampling_history, max_sample_count, relax=1):
    """次数过滤，不修改 sampling_history。
    relax>1 时（稀有桶）上限放大为 max_sample_count*relax；若仍全部达到上限，
    则放宽到硬上限 max_sample_count*3 再过滤（不清零计数，避免波及其他桶）；
    超过硬上限的候选真正排除，防止稀有样例被无限重复使用。"""
    if not sampling_history:
        return list(items)
    limit = max_sample_count * relax
    kept = [x for x in items if sampling_history.get(key_fn(x), 0) < limit]
    if kept or not items or relax <= 1:
        return kept
    hard_limit = max_sample_count * 3
    kept = [x for x in items if sampling_history.get(key_fn(x), 0) < hard_limit]
    print(f"   >>> [稀有桶放宽] {len(items)} 个候选均达上限({limit})，放宽至硬上限({hard_limit})，剩余 {len(kept)}")
    return kept


def sample_haserror_sentences(candidates, k, sampling_history=None, max_sample_count=3,
                              prediction_history=None, enable_entropy_filter=False, round_idx=1,
                              min_per_label=2, label_rarity=None):
    """
    句子层面采样：
    1. 次数过滤：按原句统计，已被采样 >= max_sample_count 次的句子剔除
    2. 句子熵：该句所有错误的 error_entropy 的平均值
    3. 最低配额：每个 golden label 至少保证 min_per_label 句（稀有类优先，总数仍不超过 k）
    4. 其余名额：entropy_curriculum_sample（无预测历史或未启用熵过滤时随机）
    """
    from utils_entropy_sampling import entropy_curriculum_sample, error_entropy

    pool = filter_by_count_with_reset(
        list(candidates), lambda c: sampling_key(c["text"]), sampling_history, max_sample_count)
    if not pool or k <= 0:
        return []
    if len(pool) <= k:
        return pool

    scores = None
    if enable_entropy_filter and prediction_history:
        scores = {}
        for c in pool:
            ents = [error_entropy(c["text"], err, prediction_history) for err in c["errors"]]
            scores[id(c)] = sum(ents) / len(ents) if ents else 0.5

    def pick(items, n):
        if n <= 0 or not items:
            return []
        if len(items) <= n:
            return list(items)
        if scores is not None:
            return entropy_curriculum_sample(items, n, lambda c: scores[id(c)], round_idx)
        return random.sample(items, n)

    # 最低配额：按 golden label 统计候选，候选少的稀有类先分配
    labels_of = {id(c): {golden_bucket_key(e) for e in c["errors"]} for c in pool}
    by_label = defaultdict(list)
    for c in pool:
        for lb in labels_of[id(c)]:
            by_label[lb].append(c)

    chosen, chosen_ids = [], set()
    # 稀有度：优先使用锚点全量错误统计的各 golden 类型句子数（未截断），无则退回当前候选数
    rarity = label_rarity or {}
    for lb in sorted(by_label, key=lambda x: (rarity.get(_norm_label(x), len(by_label[x])), len(by_label[x]))):
        if len(chosen) >= k:
            break
        covered = sum(1 for c in chosen if lb in labels_of[id(c)])
        need = min(min_per_label - covered, k - len(chosen))
        if need <= 0:
            continue
        rest = [c for c in by_label[lb] if id(c) not in chosen_ids]
        for c in pick(rest, need):
            chosen.append(c)
            chosen_ids.add(id(c))

    remaining = [c for c in pool if id(c) not in chosen_ids]
    chosen += pick(remaining, k - len(chosen))
    return chosen


def get_haserror_samples(y_true, all_error_entities_w_label, max_per_class, exclude_texts=None,
                         sampling_history=None, max_sample_count=3,
                         prediction_history=None, enable_entropy_filter=False, round_idx=1,
                         label_rarity=None):
    """
    错误 -> 归并到原句 -> 句子层面熵采样 + 次数过滤
    返回 {原句: [错误元组, ...]}，与 get_full_entity_examples_from_errors 的输入格式一致
    """
    candidates = get_haserror_candidates(y_true, all_error_entities_w_label, exclude_texts)
    sampled = sample_haserror_sentences(
        candidates, max_per_class, sampling_history, max_sample_count,
        prediction_history, enable_entropy_filter, round_idx, label_rarity=label_rarity)
    return {c["text"]: c["errors"] for c in sampled}


def _precompute_entropy_cache(all_error_text, prediction_history, enable_entropy_filter):
    """
    预计算所有候选样本的熵值，避免重复计算

    Args:
        all_error_text: 所有错误文本字典
        prediction_history: 预测历史
        enable_entropy_filter: 是否启用熵过滤

    Returns:
        entropy_cache: {(text, error_tuple): entropy_value}
    """
    if not enable_entropy_filter or not prediction_history:
        return {}

    from utils_entropy_sampling import error_entropy

    entropy_cache = {}
    for text, errors in all_error_text.items():
        for error in errors:
            key = (text, error)
            entropy_cache[key] = error_entropy(text, error, prediction_history)

    return entropy_cache


def _sample_error_category(label_needs, max_per_class, exclude_samples,
                           sample_fn, category_name="error"):
    """
    统一的错误类型采样逻辑，消除重复代码

    Args:
        label_needs: {label: [(text, error), ...]}
        max_per_class: 每个类别最多采样数量
        exclude_samples: 要排除的样本集合
        sample_fn: 采样函数 (samples, k) -> selected_samples
        category_name: 类别名称（用于日志）

    Returns:
        (sampled_dict, sampled_keys_set)
    """
    sampled_items = []
    sampled_keys = set()

    for label, samples in label_needs.items():
        # 过滤排除样本
        filtered_samples = samples
        if exclude_samples:
            filtered_samples = [(text, error) for text, error in samples
                              if (text, error) not in exclude_samples]

        # 采样
        k = min(len(filtered_samples), max_per_class)
        sampled = sample_fn(filtered_samples, k, label)

        sampled_items.extend(sampled)
        sampled_keys.update(sampled)

    return convert_to_text_errors(sampled_items), sampled_keys


def _topup_empty_category(is_empty, all_error_text, exclude_samples,
                         topup_max_per_class, sample_fn, category_name):
    """
    兜底采样：当某类错误为空时从全部错误中补充

    Args:
        is_empty: 是否为空
        all_error_text: 全部错误文本
        exclude_samples: 要排除的样本
        topup_max_per_class: 兜底采样数量
        sample_fn: 采样函数
        category_name: 类别名称

    Returns:
        (topup_dict, topup_keys_set) or (None, None)
    """
    if not is_empty:
        return None, None

    # 从全部错误中按标签类型分层采样
    label_needs_topup = defaultdict(list)
    for text, errors in all_error_text.items():
        for error in errors:
            label_needs_topup[golden_bucket_key(error)].append((text, error))

    topup_samples = []
    for label, samples in label_needs_topup.items():
        filtered_samples = samples
        if exclude_samples:
            filtered_samples = [(text, error) for text, error in samples
                              if (text, error) not in exclude_samples]
        sampled = sample_fn(filtered_samples, min(len(filtered_samples), topup_max_per_class), label)
        topup_samples.extend(sampled)

    print(f"触发兜底采样：{category_name}, 采样数量={len(topup_samples)}")
    return convert_to_text_errors(topup_samples), set(topup_samples)


def golden_bucket_key(error):
    """分桶键：优先取错误元组里的 'golden label: x'（类型混淆与漏召回统一），取不到退回 error[0]。
    这样 '锚点是 researcher、被预测成 person' 的错误进 researcher 桶而不是 person 桶。"""
    for f in error:
        if str(f).startswith("golden label:"):
            return str(f).strip()
    return error[0]


def sampling_key(text):
    """采样计数的统一键：去掉 'content: ' 前缀，使同一原句在单个专家的计数里落到同一个键。
    三个专家的计数各自独立（见 main 中 sampling_history_comp/parse/exam），仅熵共用。"""
    text = str(text)
    return text[len("content: "):] if text.startswith("content: ") else text


def stratified_sample_errors(ENTITY_TYPES, y_true, y_pred, max_per_class=50,
                             exclude_samples=None, exclude_texts=None, enable_topup=True,
                             prediction_history=None, enable_entropy_filter=False, round_idx=None,
                             sampling_history=None, max_sample_count=3):
    """
    分层采样错误样本（重构版）

    主要改进：
    1. 预计算熵缓存，避免重复计算
    2. 提取通用采样函数，消除代码重复
    3. 调整采样数量：正确样本增加到 max(2, max_per_class//3)
    4. 调整兜底采样数量：从 max_per_class//4 增加到 max(3, max_per_class//2)
    5. 硬性采样上限：每个样本最多被采样 max_sample_count 次（默认3次）

    Args:
        sampling_history: {text: 已采样次数}
        max_sample_count: 单个样本最多采样次数，默认3次
    """
    from utils_entropy_sampling import entropy_curriculum_sample, error_entropy

    # round_idx: 这批样本将用于第几轮专家更新；None 时按预热期处理（只取低熵）
    entropy_round_idx = 1 if round_idx is None else round_idx

    # 收集所有错误
    predict_error = pred_errors_all(ENTITY_TYPES, y_true, y_pred)
    recall_error = recall_errors_all(ENTITY_TYPES, y_true, y_pred)
    all_error = all_errors(predict_error, recall_error)
    correct_entities = correct_all(ENTITY_TYPES, y_true, y_pred)

    predict_error_by_text = predict_error["predict_error_by_text"]
    recall_error_text = recall_error["recall_error_text"]
    correct_text = correct_entities["correct_by_text"]
    all_error_text = all_error["all_error_text"]

    predict_error_entities_w_label_full = [(text, list(set(errors))) for text, errors in predict_error_by_text.items()]
    recall_error_entities_w_label_full = [(text, list(set(errors))) for text, errors in recall_error_text.items()]
    all_error_entities_w_label_full = [(text, list(set(errors))) for text, errors in all_error_text.items()]
    correct_entities_w_label_full = [(text, list(set(corrects))) for text, corrects in correct_text.items()]

    # 稀有度：基于锚点（伪标签 y_true）中各类型的实体数，不依赖真实标签
    label_rarity = Counter()
    for item in y_true:
        for ent in item.get("entities", []):
            if isinstance(ent, dict) and "type" in ent:
                label_rarity[str(ent["type"]).strip().lower()] += 1
    label_rarity = dict(label_rarity)
    _cnts = sorted(label_rarity.values())
    _median = _cnts[len(_cnts) // 2] if _cnts else 0
    rare_labels = {lb for lb, c in label_rarity.items() if c <= 0.5 * _median}

    # 桶（实体类型）级别：先按该专家的次数过滤，再做熵课程采样（未开熵过滤时随机）
    def bucket_filter_sample(samples, k, label=None):
        samples = filter_by_count_with_reset(
            samples, lambda s: sampling_key(s[0]), sampling_history, max_sample_count,
            relax=2 if _norm_label(label) in rare_labels else 1)
        if not samples or k <= 0:
            return []
        if len(samples) <= k:
            return list(samples)
        if enable_entropy_filter and prediction_history:
            return entropy_curriculum_sample(
                samples, k,
                lambda s: error_entropy(s[0], s[1], prediction_history),
                entropy_round_idx)
        return random.sample(samples, k)

    # 分类型追踪采样键
    sampled_keys_predict = set()
    sampled_keys_recall = set()
    sampled_keys_all = set()
    sampled_keys_correct = set()

    # ========== 改进：调整正确样本采样数量 ==========
    # 从固定的 2 个增加到 max(2, max_per_class // 3)
    correct_sample_size = max(2, max_per_class // 3)

    # ========== 代码重构：使用统一的采样函数 ==========

    # 1. 正确样本采样
    label_needs_correct = defaultdict(list)
    for text, corrects in correct_entities_w_label_full:
        for correct in corrects:
            label_needs_correct[correct[0]].append((text, correct))

    correct_entities_w_label, sampled_keys_correct = _sample_error_category(
        label_needs_correct, correct_sample_size, exclude_samples,
        bucket_filter_sample, "correct"
    )

    # 2. Predict错误采样
    label_needs_predict = defaultdict(list)
    for text, errors in predict_error_entities_w_label_full:
        for error in errors:
            label_needs_predict[golden_bucket_key(error)].append((text, error))

    predict_error_entities_w_label, sampled_keys_predict = _sample_error_category(
        label_needs_predict, max_per_class, exclude_samples,
        bucket_filter_sample, "predict_error"
    )

    # 3. Recall错误采样
    label_needs_recall = defaultdict(list)
    for text, errors in recall_error_entities_w_label_full:
        for error in errors:
            label_needs_recall[error[0]].append((text, error))

    recall_error_entities_w_label, sampled_keys_recall = _sample_error_category(
        label_needs_recall, max_per_class, exclude_samples,
        bucket_filter_sample, "recall_error"
    )

    # 4. All错误采样
    label_needs_all_error = defaultdict(list)
    for text, errors in all_error_entities_w_label_full:
        for error in errors:
            label_needs_all_error[golden_bucket_key(error)].append((text, error))

    # 按实体类型采样错误：先做次数过滤，再随机采样（不做熵过滤）
    def count_filter_random_sample(samples, k, label=None):
        samples = filter_by_count_with_reset(
            samples, lambda s: sampling_key(s[0]), sampling_history, max_sample_count,
            relax=2 if _norm_label(label) in rare_labels else 1)
        if not samples or k <= 0:
            return []
        return list(samples) if len(samples) <= k else random.sample(samples, k)

    all_error_entities_w_label, sampled_keys_all = _sample_error_category(
        label_needs_all_error, max_per_class, exclude_samples,
        count_filter_random_sample, "all_error"
    )

    # 还原为完整标注句（带错误信息），再在句子层面做熵采样 + 次数过滤
    all_error_entities_w_label_gold = get_haserror_samples(
        y_true, all_error_entities_w_label, max_per_class, exclude_texts,
        sampling_history=sampling_history, max_sample_count=max_sample_count,
        prediction_history=prediction_history, enable_entropy_filter=enable_entropy_filter,
        round_idx=entropy_round_idx, label_rarity=label_rarity
    )

    sampled_keys_all_gold = set(all_error_entities_w_label_gold.keys())

    # ========== 空集兜底补充 ==========
    if enable_topup:
        # 改进：兜底采样数量从 max_per_class//4 增加到 max(3, max_per_class//2)
        topup_max_per_class = max(3, max_per_class // 2)

        # 1) predict_error 兜底
        result = _topup_empty_category(
            not predict_error_entities_w_label,
            all_error_text, exclude_samples, topup_max_per_class,
            bucket_filter_sample, "predict_error_entities_w_label"
        )
        if result[0] is not None:
            predict_error_entities_w_label, sampled_keys_predict = result

        # 2) recall_error 兜底
        result = _topup_empty_category(
            not recall_error_entities_w_label,
            all_error_text, exclude_samples, topup_max_per_class,
            bucket_filter_sample, "recall_error_entities_w_label"
        )
        if result[0] is not None:
            recall_error_entities_w_label, sampled_keys_recall = result

        # 3) all_error_gold 兜底
        if not all_error_entities_w_label_gold:
            # 从全部错误还原完整标注句（带错误信息），同样做句子级熵采样 + 次数过滤
            topup_gold = get_haserror_samples(
                y_true, all_error_text, topup_max_per_class, exclude_texts,
                sampling_history=sampling_history, max_sample_count=max_sample_count,
                prediction_history=prediction_history, enable_entropy_filter=enable_entropy_filter,
                round_idx=entropy_round_idx, label_rarity=label_rarity
            )
            all_error_entities_w_label_gold = topup_gold
            sampled_keys_all_gold = set(topup_gold.keys())
            print(f"触发兜底采样：all_error_entities_w_label_gold, 采样数量={len(sampled_keys_all_gold)}")

    return {
        "predict_error_entities_w_label": predict_error_entities_w_label,
        "recall_error_entities_w_label": recall_error_entities_w_label,
        "all_error_entities_w_label": all_error_entities_w_label,
        "all_error_entities_w_label_gold": all_error_entities_w_label_gold,

        "predict_error_entities_w_label_full": predict_error_entities_w_label_full,
        "recall_error_entities_w_label_full": recall_error_entities_w_label_full,
        "all_error_entities_w_label_full": all_error_entities_w_label_full,

        "sampled_keys_predict": sampled_keys_predict,
        "sampled_keys_recall": sampled_keys_recall,
        "sampled_keys_all": sampled_keys_all,
        "sampled_keys_all_gold": sampled_keys_all_gold,
    }


def convert_label(datas, dataset):
    outs = []
    for data in datas:
        entities = data["entities"]
        new_entities = []
        for ent in entities:
            new_ent = ent.copy()
            new_entities.append(new_ent)
        new_data = data.copy()
        new_data["entities"] = new_entities
        outs.append(new_data)
    return outs


def get_full_entity_examples_from_errors(y_true, error_samples, max_samples=10, with_error_context=True):
    """
    从错误样本中提取完整的实体列表作为案例展示，并附带错误上下文（Missing / Confusion）。

    返回格式示例：
        Context: "原文本内容..."
            Answer: [{"type": "疾病", "entity": "流感"}, {"type": "临床表现", "entity": "发烧"}, ...]
            Error Context:
            - Missing: entity="流感" should be labeled as "疾病"
            - Confusion: entity="发烧" (labeled as "疾病") should be labeled as "临床表现"

    Args:
        y_true: 真实标注数据 [{"text": "...", "entities": [...]}, ...]
        error_samples: 错误样本 {text_key: [error_tuples_or_lists, ...], ...}
            - Confusion 项（来自 pred_errors_all）:
              ("predict label: <type>", "entity: <name>", "golden label: <type>")
            - Missing 项（来自 recall_errors_all）:
              ("golden label: <type>", "entity: <name>", "Easily missed entity")
        max_samples: 最多返回多少个案例
        with_error_context: 是否附带错误上下文；False 时只保留 Context + 完整正确标注（供 example 专家使用）

    Returns:
        格式化的完整实体案例字符串（多个案例以换行拼接）
    """

    def _parse_kv_string(item):
        """解析形如 'predict label: xxx' / 'entity: xxx' 的字符串为 (key, value)"""
        if ":" in item:
            k, v = item.split(":", 1)
            return k.strip(), v.strip()
        return None, item

    def _classify_error(error_item):
        """
        区分错误类型：
        - Missing: 召回错误（真实标签未识别），eval_type 通常含 'Easily missed entity'，
                    或者没有 'predict label' 但有 'golden label'
        - Confusion: 预测类型错误，eval_type 一般同时包含 'predict label' 与 'golden label'
        返回 (category, entity, predict_label, golden_label, raw_desc)
        """
        # error_item 可能是 tuple / list / str，统一成 list[str]
        if isinstance(error_item, (tuple, list)):
            fields = list(error_item)
        else:
            fields = [str(error_item)]

        kv_map = {}
        for f in fields:
            k, v = _parse_kv_string(str(f))
            if k:
                kv_map[k] = v

        entity = kv_map.get("entity", "")
        predict_label = kv_map.get("predict label", None)
        golden_label = kv_map.get("golden label", None)

        # 判定是否为 Missing：描述中包含 'Easily missed entity' 或没有 predict label
        raw_desc = " ".join(str(f) for f in fields)
        if "Easily missed entity" in raw_desc or predict_label is None:
            category = "Missing"
        else:
            category = "Confusion"

        return category, entity, predict_label, golden_label, raw_desc

    # 构建 y_true 的快速查找字典
    y_true_dict = {}
    for item in y_true:
        y_true_dict[item["text"]] = item.get("entities", [])

    def _round_robin_select(items, limit):
        """按 golden label 轮流取句子：候选少的标签先取，已选句子不重复，凑够 limit 为止。"""
        if len(items) <= limit:
            return list(items)
        by_label = defaultdict(list)
        for idx, (_, errs) in enumerate(items):
            for lb in {golden_bucket_key(e) for e in errs} or {None}:
                by_label[lb].append(idx)
        order = sorted(by_label, key=lambda lb: len(by_label[lb]))
        picked, picked_set = [], set()
        cursors = {lb: 0 for lb in order}
        while len(picked) < limit:
            progressed = False
            for lb in order:
                lst = by_label[lb]
                while cursors[lb] < len(lst) and lst[cursors[lb]] in picked_set:
                    cursors[lb] += 1
                if cursors[lb] < len(lst):
                    idx = lst[cursors[lb]]
                    picked.append(idx)
                    picked_set.add(idx)
                    progressed = True
                    if len(picked) >= limit:
                        break
            if not progressed:
                break
        return [items[i] for i in sorted(picked)]

    full_examples = []
    for text_key, errors in _round_robin_select(list(error_samples.items()), max_samples):
        # 去掉 "content: " 前缀
        clean_text = text_key.replace("content: ", "")

        # 在 y_true 中查找完整的实体列表
        if clean_text in y_true_dict:
            full_entities = y_true_dict[clean_text]

            # 格式化为目标输出格式
            entities_json = json.dumps(full_entities, ensure_ascii=False)

            # 解析错误上下文
            error_lines = []
            seen_signatures = set()  # 用于去重相同的错误描述
            for err in (errors if with_error_context else []):
                category, entity, p_label, g_label, _ = _classify_error(err)
                if category == "Missing":
                    line = f'- Missing: entity="{entity}" should be labeled as "{g_label}"'
                else:  # Confusion
                    line = f'- Confusion: entity="{entity}" (labeled as "{p_label}") should be labeled as "{g_label}"'
                # 去重
                if line not in seen_signatures:
                    seen_signatures.add(line)
                    error_lines.append(line)

            if error_lines:
                error_block = "Error Context:\n    " + "\n    ".join(error_lines)
                formatted_example = (
                    f'Context: "{clean_text}"\n'
                    f'    Answer: {entities_json}\n'
                    f'    {error_block}'
                )
            else:
                formatted_example = (
                    f'Context: "{clean_text}"\n'
                    f'    Answer: {entities_json}'
                )

            full_examples.append(formatted_example)
    return "\n\n".join(full_examples) if full_examples else "No examples available"
