"""
指令样例精简模块（Example Selection）

每轮专家优化后，新指令中的 "## Entity Examples" / "## Annotation Examples" 段落
可能随迭代不断膨胀。本模块对其做后处理：
    1. 解析出全部样例（兼容两种格式）
         - 1. Context: "..." -> Span: "x" is identified as T; Span: "y" is identified as T2; ...
         - 1. "..." -> Answer: [{"type": "T", "entity": "x"}, ...]
    2. 按上下文文本去重
    3. 以 [文本 TF-IDF ; 标签 multi-hot] 为特征做 KMeans 聚类（保证多样性），簇数 = example_num，
       每簇取"代价"最小的样例作为代表：代价 = 预测熵 + DIST_WEIGHT * 归一化的离质心距离
       （预测熵取自 prediction_history，越低说明该上下文的预测越稳定、标注越可信；
        未提供历史时熵恒定，退化为取离质心最近的样例）
    4. 标签覆盖修复：若代表样例未覆盖当前样例中出现的全部标签，
       优先在簇内做"净增覆盖"替换；替换不可行时追加额外样例（覆盖优先于数量），
       候选之间同样优先选代价低（低熵）的样例
    5. 按原始顺序重新编号写回指令
"""
import json
import re
import warnings

# 标签特征相对文本特征的权重（两者均已 L2 归一化）
LABEL_FEATURE_WEIGHT = 1.0
# 簇内选代表时，离质心距离（簇内归一化到 [0,1]）相对预测熵（[0,1]）的权重。
# 越小越偏向低熵，越大越偏向典型（靠近质心）样例。
DIST_WEIGHT = 0.5
# 是否强制覆盖所有标签：True 时保证选中样例覆盖全部标签（可能超出 example_num），
# False 时仅选聚类代表样例（数量严格等于 example_num，可能有标签缺失）
ENFORCE_LABEL_COVERAGE = True

_SECTION_HEADER_RE = re.compile(
    r'^##\s*(?:(?:Entity|Annotation)\s+Examples?|标注示例|实体示例|示例|样例)\b[^\n]*$',
    re.I | re.M)
# 样例段落结束于下一个 "##" 标题，或模板中的 "Notes:" / "注意:" 段落
_NEXT_HEADER_RE = re.compile(r'^(?:##\s|\s*(?:Notes|注意事项|注意)\s*[:：])', re.M | re.I)
_ITEM_START_RE = re.compile(r'^\s*(?:[-*•]\s+|\d+\s*[.)、]\s*)')
_ITEM_MARKER_RE = re.compile(r'^\s*(?:[-*•]\s*)?(?:\d+\s*[.)、]\s*)?')
_TEXT_RE = re.compile(r'^\s*(?:Context\s*[:：]\s*)?["“](.*)["”]\s*(?:->|→)', re.S)
_SPAN_LABEL_RE = re.compile(r'is\s+identified\s+as\s+([^;\n]+)', re.I)
_JSON_TYPE_RE = re.compile(r'["\']type["\']\s*:\s*["\']([^"\']+)["\']')
_LABEL_STRIP_RE = re.compile(r'^[\s\[\]()*"\'`“”.。,，;；]+|[\s\[\]()*"\'`“”.。,，;；]+$')
_CJK_RE = re.compile(r'[一-鿿]')


def _norm_label(label):
    return _LABEL_STRIP_RE.sub('', label).strip().lower()


def _parse_single_example(body):
    """从单条样例中解析上下文文本与标签集合。"""
    m = _TEXT_RE.match(body)
    if m:
        text = m.group(1).strip()
    else:
        text = re.split(r'->|→', body, maxsplit=1)[0].strip()
    # 标签只从箭头右侧解析，避免上下文文本中的偶然匹配
    parts = re.split(r'->|→', body, maxsplit=1)
    answer_part = parts[1] if len(parts) > 1 else body
    labels = set()
    for raw in _SPAN_LABEL_RE.findall(answer_part) + _JSON_TYPE_RE.findall(answer_part):
        label = _norm_label(raw)
        if label:
            labels.add(label)
    return {"body": body, "text": text, "labels": frozenset(labels)}


def parse_examples(section_text):
    """将样例段落切分为样例列表；非编号开头的行视为上一条样例的续行。"""
    bodies = []
    for line in section_text.splitlines():
        if not line.strip():
            continue
        if _ITEM_START_RE.match(line) or not bodies:
            bodies.append(_ITEM_MARKER_RE.sub('', line, count=1).strip())
        else:
            bodies[-1] += " " + line.strip()
    examples = [_parse_single_example(b) for b in bodies if b]

    # 按上下文文本去重（保留首次出现）
    seen, unique = set(), []
    for ex in examples:
        key = re.sub(r'\s+', ' ', ex["text"]).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(ex)
    return unique


def _build_features(examples, all_labels):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = [ex["text"] or ex["body"] for ex in examples]
    if any(_CJK_RE.search(t) for t in texts):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    else:
        vectorizer = TfidfVectorizer(analyzer="word", stop_words="english", ngram_range=(1, 2))
    try:
        x_text = vectorizer.fit_transform(texts)
    except ValueError:  # 词表为空（文本过短/全是停用词）时退化为字符 n-gram
        x_text = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts)
    x_text = normalize(x_text).toarray()

    label_index = {label: i for i, label in enumerate(sorted(all_labels))}
    x_label = np.zeros((len(examples), max(1, len(label_index))))
    for row, ex in enumerate(examples):
        for label in ex["labels"]:
            x_label[row, label_index[label]] = 1.0
    x_label = normalize(x_label)
    return np.hstack([x_text, LABEL_FEATURE_WEIGHT * x_label])


def compute_example_entropies(examples, prediction_history):
    """
    每条样例的句子级预测熵 [0,1]：用样例上下文在 prediction_history 中查找。
    无历史 / 未匹配 / 历史不足 2 条时取 0.5（中性值）。

    改进：使用全局预测池（包含所有专家的预测）+ 智能相似度匹配回退

    匹配策略（三层回退）：
    1. 精确匹配：text 完全一致
    2. 归一化匹配：去除空格、标点后匹配
    3. 相似度匹配：找最相似的样例（避免LLM改写导致无法匹配）

    优势：
    - 新样例在当前轮也有预测（compare/parse/example/fusion 共4个）
    - 基于真实专家分歧计算熵，无需人工打折
    - 充分利用跨专家信息
    - 容错性强，处理文本改写情况
    """
    if not prediction_history:
        return [0.5] * len(examples)
    try:
        from utils_entropy_sampling import sentence_entropy
    except Exception:
        return [0.5] * len(examples)

    # 构建归一化文本映射
    norm_map = {}
    for text in prediction_history:
        norm_map.setdefault(re.sub(r'\s+', ' ', str(text)).strip().lower(), text)

    # 统计匹配情况
    match_stats = {
        "exact_match": 0,
        "normalized_match": 0,
        "similarity_match": 0,
        "no_match": 0
    }

    # 直接匹配失败的样例：(类型, 样例原文, 匹配到的文本, 相似度)
    fallback_logs = []

    result = []
    for ex in examples:
        text = ex["text"]
        matched_text = None
        match_type = None

        # 策略1：精确匹配
        if text in prediction_history:
            matched_text = text
            match_type = "exact_match"
        else:
            # 策略2：归一化匹配
            normalized_text = re.sub(r'\s+', ' ', text).strip().lower()
            if normalized_text in norm_map:
                matched_text = norm_map[normalized_text]
                match_type = "normalized_match"
            else:
                # 策略3：相似度匹配（兜底）
                matched_text, sim_score = _find_most_similar_text(
                    text, prediction_history, return_score=True)
                if matched_text:
                    match_type = "similarity_match"
                else:
                    match_type = "no_match"
                fallback_logs.append((match_type, text, matched_text, sim_score))

        # 更新统计
        match_stats[match_type] += 1

        # 计算熵
        if matched_text:
            entropy = sentence_entropy(matched_text, prediction_history)
        else:
            # 理论上不应该到这里（相似度匹配应该总能找到）
            entropy = 0.5

        result.append(entropy)

    # 输出匹配统计
    total = len(examples)
    if total > 0:
        print(f"[EntropyMatch] 精确:{match_stats['exact_match']}/{total} "
              f"归一化:{match_stats['normalized_match']}/{total} "
              f"相似度:{match_stats['similarity_match']}/{total} "
              f"未匹配:{match_stats['no_match']}/{total}")
        direct_match = match_stats['exact_match'] + match_stats['normalized_match']
        match_rate = direct_match / total * 100
        total_match = direct_match + match_stats['similarity_match']
        total_rate = total_match / total * 100
        print(f"[EntropyMatch] 直接匹配率:{match_rate:.1f}% 总匹配率:{total_rate:.1f}%")

        # 打印直接匹配失败、经相似度回退才匹配上（或仍未匹配）的样例
        for match_type, query, matched, score in fallback_logs:
            if match_type == "similarity_match":
                print(f"[EntropyMatch] 直接匹配失败→相似度匹配成功 (sim={score:.3f})\n"
                      f"    样例  : {query}\n"
                      f"    匹配到: {matched}")
            else:
                print(f"[EntropyMatch] 直接匹配失败→仍未匹配 (熵取0.5)\n"
                      f"    样例  : {query}")

    return result


def _find_most_similar_text(query_text, candidate_texts, threshold=0.3, return_score=False):
    """
    找到最相似的文本（基于字符级相似度）

    Args:
        query_text: 查询文本
        candidate_texts: 候选文本列表（prediction_history的keys）
        threshold: 最低相似度阈值（低于此值返回None）
        return_score: True 时返回 (最相似文本, 相似度)，未命中为 (None, 0.0)

    Returns:
        最相似的文本，如果所有相似度都低于阈值则返回None
    """
    if not candidate_texts:
        return (None, 0.0) if return_score else None

    best_match = None
    best_score = threshold

    # 预处理查询文本
    query_normalized = re.sub(r'\s+', '', query_text.lower())
    query_len = len(query_normalized)

    for candidate in candidate_texts:
        # 预处理候选文本
        candidate_normalized = re.sub(r'\s+', '', str(candidate).lower())
        candidate_len = len(candidate_normalized)

        # 快速过滤：长度差异过大直接跳过
        if abs(query_len - candidate_len) > max(query_len, candidate_len) * 0.5:
            continue

        # 计算字符级相似度（Jaccard相似度，快速）
        # 使用字符2-gram
        query_bigrams = set(query_normalized[i:i+2] for i in range(len(query_normalized)-1))
        candidate_bigrams = set(candidate_normalized[i:i+2] for i in range(len(candidate_normalized)-1))

        if not query_bigrams or not candidate_bigrams:
            continue

        intersection = len(query_bigrams & candidate_bigrams)
        union = len(query_bigrams | candidate_bigrams)
        similarity = intersection / union if union > 0 else 0

        if similarity > best_score:
            best_score = similarity
            best_match = candidate

    if return_score:
        return best_match, (best_score if best_match is not None else 0.0)
    return best_match


_SPAN_TEXT_RE = re.compile(r'Span\s*:\s*["“](.+?)["”]\s+is\s+identified', re.I)
_JSON_ENTITY_RE = re.compile(r'["\']entity["\']\s*:\s*["\']([^"\']+)["\']')
# 样例上下文与训练文本的相似度低于该值，视为模型自行生成（非训练数据）
PROVENANCE_SIM_THRESHOLD = 0.6
# 非原文（改写 + 自造）样例占比超过该值时打印预警
PROVENANCE_WARN_RATE = 0.2


def _compact(text):
    """去掉空白与标点并转小写，消除模型改写空格/标点带来的假差异。"""
    return re.sub(r'[\W_]+', '', str(text).lower())


def audit_example_provenance(examples, known_texts, logger=print):
    """
    溯源监控：检查指令中的样例上下文是否来自训练数据（known_texts）。
    分类：
      verbatim   忽略空白/标点后与某条训练文本一致
      excerpt    是某条训练文本的片段
      rewritten  与某条训练文本高度相似但内容被改写
      fabricated 找不到相似训练文本，疑似模型自行生成
    另检查样例中标注的实体片段是否真实出现在其上下文中。
    发现改写/自造样例或片段不在上下文中时打印预警。
    """
    if not examples or not known_texts:
        return None
    known = list(known_texts)
    known_keys = {}
    for t in known:
        known_keys.setdefault(_compact(t), t)

    counts = {"verbatim": 0, "excerpt": 0, "rewritten": 0, "fabricated": 0}
    suspicious, bad_span = [], []
    drop_idx = set()  # 改写 / 自造 / 实体不在上下文的样例下标，当前轮次不使用
    for idx, ex in enumerate(examples):
        text = ex["text"] or ex["body"]
        key = _compact(text)
        if key in known_keys:
            kind = "verbatim"
        elif key and any(key in k for k in known_keys):
            kind = "excerpt"
        else:
            sim = _find_most_similar_text(text, known, threshold=PROVENANCE_SIM_THRESHOLD)
            kind = "rewritten" if sim else "fabricated"
        counts[kind] += 1
        if kind in ("rewritten", "fabricated"):
            suspicious.append((kind, text))
            drop_idx.add(idx)

        answer = re.split(r'->|→', ex["body"], maxsplit=1)
        answer = answer[1] if len(answer) > 1 else ""
        spans = _SPAN_TEXT_RE.findall(answer) + _JSON_ENTITY_RE.findall(answer)
        missing = [s for s in spans if _compact(s) and _compact(s) not in key]
        if missing:
            bad_span.append((text, missing))
            drop_idx.add(idx)

    total = len(examples)
    bad = counts["rewritten"] + counts["fabricated"]
    logger(f"[ExampleProvenance] 共 {total} 条: 原文 {counts['verbatim']}, 片段 {counts['excerpt']}, "
           f"改写 {counts['rewritten']}, 自造 {counts['fabricated']}, 实体不在上下文 {len(bad_span)}")
    if bad / total > PROVENANCE_WARN_RATE or counts["fabricated"] > 0 or bad_span:
        logger(f"   ⚠️ 预警: 非训练原文样例占比 {bad / total:.0%}（改写 {counts['rewritten']}, "
               f"自造 {counts['fabricated']}），实体不在上下文 {len(bad_span)} 条")
        for kind, text in suspicious[:3]:
            logger(f"      [{kind}] {text[:80]}...")
        for text, missing in bad_span[:3]:
            logger(f"      [实体不在上下文] {missing[:3]} <- {text}")
    return {**counts, "bad_span": len(bad_span), "total": total, "drop_idx": drop_idx}


def _greedy_cover_fallback(examples, k, all_labels):
    """无 sklearn 或聚类失败时的兜底：贪心集合覆盖 + 按原顺序补足到 k。"""
    selected, covered = [], set()
    while covered != all_labels:
        best = max((i for i in range(len(examples)) if i not in selected),
                   key=lambda i: len(examples[i]["labels"] - covered), default=None)
        if best is None or not (examples[best]["labels"] - covered):
            break
        selected.append(best)
        covered |= examples[best]["labels"]
    for i in range(len(examples)):
        if len(selected) >= k:
            break
        if i not in selected:
            selected.append(i)
    return sorted(selected)


def select_representative_examples(examples, k, random_state=24, entropies=None,
                                   enforce_coverage=None):
    """
    返回被选中样例的下标（按原顺序）。
    enforce_coverage: 是否强制覆盖全部标签（None 时使用 ENFORCE_LABEL_COVERAGE）。
      - True: 选中样例的标签并集 == 全部样例的标签并集，数量可能 > k
      - False: 严格选 k 个聚类代表样例，可能有标签缺失
    entropies: 每条样例的预测熵 [0,1]（None 时视为恒定，不影响选择）。
    """
    if enforce_coverage is None:
        enforce_coverage = ENFORCE_LABEL_COVERAGE

    n = len(examples)
    if k <= 0 or n <= k:
        return list(range(n))
    labels = [ex["labels"] for ex in examples]
    all_labels = set().union(*labels)

    try:
        import numpy as np
        from sklearn.cluster import KMeans

        x = _build_features(examples, all_labels)
        with warnings.catch_warnings():
            # Windows + MKL 下 KMeans 会对小样本反复输出内存泄漏警告，屏蔽以免刷屏
            warnings.filterwarnings("ignore", message=".*memory leak on Windows with MKL.*")
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(x)
        cluster_ids = km.labels_
        dist = np.linalg.norm(x - km.cluster_centers_[cluster_ids], axis=1)
    except Exception as e:
        print(f"[ExampleSelect] 聚类失败，使用贪心覆盖兜底: {e}")
        return _greedy_cover_fallback(examples, k, all_labels)

    if entropies is None:
        entropies = [0.5] * n
    # 代价 = 预测熵 + DIST_WEIGHT * 簇内归一化离质心距离；越小越优先（低熵且较典型）
    cluster_max = {}
    for i in range(n):
        c = int(cluster_ids[i])
        cluster_max[c] = max(cluster_max.get(c, 0.0), float(dist[i]))
    cost = [entropies[i] + DIST_WEIGHT * (float(dist[i]) / cluster_max[int(cluster_ids[i])]
                                          if cluster_max[int(cluster_ids[i])] > 0 else 0.0)
            for i in range(n)]

    # 每簇取代价最小的样例作为代表（簇内多样性由聚类保证）
    reps = {}
    for i in range(n):
        c = int(cluster_ids[i])
        if c not in reps or cost[i] < cost[reps[c]]:
            reps[c] = i

    if not enforce_coverage:
        # 不强制覆盖：只返回聚类代表样例
        return sorted(reps.values())

    extras = []

    def covered_by(exclude=None):
        idxs = [i for i in list(reps.values()) + extras if i != exclude]
        return set().union(*(labels[i] for i in idxs)) if idxs else set()

    uncovered = all_labels - covered_by()
    while uncovered:
        chosen = set(reps.values()) | set(extras)
        best, best_score = None, None
        for i in range(n):
            if i in chosen:
                continue
            gain = labels[i] & uncovered
            if not gain:
                continue
            c = int(cluster_ids[i])
            cur = reps.get(c)
            # 替换掉簇代表后会丢失的、仅由该代表覆盖的标签
            loss = (labels[cur] - covered_by(exclude=cur) - labels[i]) if cur is not None else set()
            net = len(gain) - len(loss)
            if net <= 0:
                continue
            score = (net, -cost[i])
            if best_score is None or score > best_score:
                best, best_score = i, score
        if best is not None:
            reps[int(cluster_ids[best])] = best
        else:
            # 簇内替换无法净增覆盖，追加一个额外样例（标签覆盖优先于数量约束）
            candidates = [i for i in range(n) if i not in chosen and labels[i] & uncovered]
            if not candidates:
                break
            extra = max(candidates, key=lambda i: (len(labels[i] & uncovered), -cost[i]))
            extras.append(extra)
        uncovered = all_labels - covered_by()

    return sorted(set(reps.values()) | set(extras))


def compress_prompt_examples(prompt_text, example_num, random_state=24, logger=print,
                             prediction_history=None, enforce_coverage=None):
    """
    对优化后的指令做样例精简；未找到样例段落或 example_num<=0 时原样返回。
    prediction_history: {text: [各轮预测实体列表]}，提供时在多样性与标签覆盖的前提下优先选低熵样例。
    enforce_coverage: 是否强制覆盖全部标签（None 时使用 ENFORCE_LABEL_COVERAGE 全局配置）。
    """
    if not prompt_text or example_num is None or example_num <= 0:
        return prompt_text
    header = _SECTION_HEADER_RE.search(prompt_text)
    if not header:
        return prompt_text
    start = header.end()
    nxt = _NEXT_HEADER_RE.search(prompt_text, start)
    end = nxt.start() if nxt else len(prompt_text)

    examples = parse_examples(prompt_text[start:end])
    if not examples:
        return prompt_text

    # 溯源监控：prediction_history 的键即全部训练文本
    audit = audit_example_provenance(examples, prediction_history.keys() if prediction_history else None, logger)

    # 丢弃改写 / 自造 / 实体不在上下文的样例，不进入聚类选样，也不写回当前轮次指令
    if audit and audit["drop_idx"]:
        n_before = len(examples)
        examples = [ex for i, ex in enumerate(examples) if i not in audit["drop_idx"]]
        logger(f"[ExampleProvenance] 已丢弃 {n_before - len(examples)}/{n_before} 条不可信样例，剩余 {len(examples)} 条")
        if not examples:
            logger("   ⚠️ 全部样例均被丢弃，本轮指令不含样例段落")
            return prompt_text[:header.start()].rstrip() + ("\n\n" + prompt_text[end:].strip() if prompt_text[end:].strip() else "")

    entropies = compute_example_entropies(examples, prediction_history)
    selected = select_representative_examples(examples, example_num, random_state=random_state,
                                              entropies=entropies, enforce_coverage=enforce_coverage)
    all_labels = set().union(*(ex["labels"] for ex in examples))
    kept_labels = set().union(*(examples[i]["labels"] for i in selected)) if selected else set()
    mean_all = sum(entropies) / len(entropies)
    mean_sel = sum(entropies[i] for i in selected) / len(selected) if selected else 0.0
    coverage_flag = "强制覆盖" if (enforce_coverage if enforce_coverage is not None else ENFORCE_LABEL_COVERAGE) else "聚类代表"
    logger(f"[ExampleSelect] 样例 {len(examples)} -> {len(selected)} (目标={example_num}, {coverage_flag})，"
           f"标签覆盖 {len(kept_labels)}/{len(all_labels)}，平均熵 {mean_all:.3f} -> {mean_sel:.3f}")

    new_section = "\n".join(f"- {rank}. {examples[i]['body']}" for rank, i in enumerate(selected, 1))
    tail = prompt_text[end:].strip()
    result = prompt_text[:start].rstrip() + "\n" + new_section
    return result + ("\n\n" + tail if tail else "")


# ======================================================================
# 样例标注随锚点刷新
# ======================================================================

_ARROW_RE = re.compile(r'->|→')
_CONTEXT_PREFIX_RE = re.compile(r'^\s*(Context\s*[:：]\s*)', re.I)


def _detect_answer_format(answer_part):
    """判断箭头右侧的标注格式：'span' | 'json' | None。"""
    if _SPAN_LABEL_RE.search(answer_part):
        return "span"
    if _JSON_TYPE_RE.search(answer_part):
        return "json"
    return None


def _clean_anchor_entities(entities):
    """锚点实体 -> [{"type", "entity"}]，去空去重并保持原顺序。"""
    out, seen = [], set()
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        etype = str(ent.get("type", "")).strip()
        value = str(ent.get("entity", "")).strip()
        if not etype or not value or (etype, value) in seen:
            continue
        seen.add((etype, value))
        out.append({"type": etype, "entity": value})
    return out


def _format_answer(entities, fmt):
    """按样例原有格式生成箭头右侧的标注文本。"""
    if fmt == "span":
        return "; ".join(f'Span: "{e["entity"]}" is identified as {e["type"]}' for e in entities)
    return "Answer: " + json.dumps(entities, ensure_ascii=False)


def _match_anchor_text(text, anchor_map, norm_map):
    """三层匹配：精确 -> 归一化 -> 相似度（阈值同溯源检查）。返回 (锚点文本或 None, 匹配类型, 相似度)。"""
    if text in anchor_map:
        return text, "exact", 1.0
    norm = re.sub(r'\s+', ' ', text).strip().lower()
    if norm in norm_map:
        return norm_map[norm], "normalized", 1.0
    matched, score = _find_most_similar_text(
        text, list(anchor_map.keys()), threshold=PROVENANCE_SIM_THRESHOLD, return_score=True)
    if matched is not None:
        return matched, "similarity", score
    return None, "none", 0.0

def refresh_examples_with_anchor(prompt_text, anchor_data, logger=print):
    """
    把指令样例段落中每条样例的标注，替换为锚点集 anchor_data 当前给出的实体。

    - 保留每条样例原有的标注格式：
        JSON : - n. "ctx" -> Answer: [{"type": "T", "entity": "x"}, ...]
        Span : - n. "ctx" -> Span: "x" is identified as T; Span: "y" is identified as T2
      （带 "Context:" 前缀的也保留前缀）
    - 样例上下文先与锚点文本做精确 / 归一化 / 相似度匹配；匹配上后上下文也改回锚点原句，
      以修正 LLM 改写。未匹配到的样例原样保留（交给 audit_example_provenance 处理）。
    - 锚点中该句没有实体时丢弃该样例（空标注在 Span 格式下无法表达，也会误导模型）。
    - 刷新后按上下文再次去重并重新编号。

    anchor_data: [{"text": str, "entities": [{"type":..., "entity":...}, ...]}, ...]
    未找到样例段落 / 锚点为空时原样返回。
    """
    if not prompt_text or not anchor_data:
        return prompt_text
    header = _SECTION_HEADER_RE.search(prompt_text)
    if not header:
        return prompt_text
    start = header.end()
    nxt = _NEXT_HEADER_RE.search(prompt_text, start)
    end = nxt.start() if nxt else len(prompt_text)

    examples = parse_examples(prompt_text[start:end])
    if not examples:
        return prompt_text

    anchor_map = {}
    for item in anchor_data:
        if isinstance(item, dict) and "text" in item:
            anchor_map.setdefault(str(item["text"]), item.get("entities", []))
    norm_map = {}
    for text in anchor_map:
        norm_map.setdefault(re.sub(r'\s+', ' ', text).strip().lower(), text)

    # 无法从自身判断格式的样例，沿用段落内出现最多的格式（都没有则用 JSON）
    fmts = []
    for ex in examples:
        parts = _ARROW_RE.split(ex["body"], maxsplit=1)
        fmts.append(_detect_answer_format(parts[1]) if len(parts) > 1 else None)
    known = [f for f in fmts if f]
    default_fmt = max(set(known), key=known.count) if known else "json"

    stats = {"updated": 0, "unchanged": 0, "unmatched": 0, "dropped_empty": 0, "dup": 0}
    new_bodies, seen_keys, changes = [], set(), []

    for ex, fmt in zip(examples, fmts):
        body = ex["body"]
        matched, mtype, score = _match_anchor_text(ex["text"], anchor_map, norm_map)
        if matched is None:
            stats["unmatched"] += 1
            key = re.sub(r'\s+', ' ', ex["text"]).strip().lower()
            if key not in seen_keys:
                seen_keys.add(key)
                new_bodies.append(body)
            continue

        entities = _clean_anchor_entities(anchor_map[matched])
        if not entities:
            stats["dropped_empty"] += 1
            changes.append(f"    丢弃(锚点无实体): {matched}")
            continue

        key = re.sub(r'\s+', ' ', matched).strip().lower()
        if key in seen_keys:
            stats["dup"] += 1
            continue
        seen_keys.add(key)

        prefix = _CONTEXT_PREFIX_RE.match(body)
        prefix_str = prefix.group(1) if prefix else ""
        new_body = f'{prefix_str}"{matched}" -> {_format_answer(entities, fmt or default_fmt)}'

        old_parts = _ARROW_RE.split(body, maxsplit=1)
        old_pairs = {(_norm_label(l), e.strip().lower()) for l, e in
                     _extract_pairs(old_parts[1] if len(old_parts) > 1 else "")}
        new_pairs = {(_norm_label(e["type"]), e["entity"].strip().lower()) for e in entities}
        if old_pairs == new_pairs and matched == ex["text"]:
            stats["unchanged"] += 1
        else:
            stats["updated"] += 1
            tag = "" if mtype in ("exact", "normalized") else f" (sim={score:.2f}, 上下文已校正)"
            changes.append(f"    更新{tag}: {matched}\n"
                           f"        旧: {sorted(old_pairs)}\n"
                           f"        新: {sorted(new_pairs)}")
        new_bodies.append(new_body)

    logger(f"[ExampleRefresh] 更新:{stats['updated']} 未变:{stats['unchanged']} "
           f"未匹配保留:{stats['unmatched']} 锚点无实体丢弃:{stats['dropped_empty']} 去重:{stats['dup']}")
    # for line in changes:
    #     logger(line)

    if not new_bodies:
        logger("   ⚠️ 刷新后无样例，本轮指令不含样例段落")
        tail = prompt_text[end:].strip()
        return prompt_text[:header.start()].rstrip() + ("\n\n" + tail if tail else "")

    new_section = "\n".join(f"- {rank}. {b}" for rank, b in enumerate(new_bodies, 1))
    tail = prompt_text[end:].strip()
    result = prompt_text[:start].rstrip() + "\n" + new_section
    return result + ("\n\n" + tail if tail else "")


_JSON_PAIR_RE = re.compile(
    r'\{[^{}]*?["\']type["\']\s*:\s*["\']([^"\']+)["\'][^{}]*?["\']entity["\']\s*:\s*["\']([^"\']+)["\'][^{}]*?\}')
_SPAN_PAIR_RE = re.compile(r'Span\s*:\s*["“](.+?)["”]\s+is\s+identified\s+as\s+([^;\n]+)', re.I)


def _extract_pairs(answer_part):
    """从箭头右侧解析 (type, entity) 对，兼容 JSON 与 Span 两种格式。"""
    pairs = [(t, e) for t, e in _JSON_PAIR_RE.findall(answer_part)]
    pairs += [(t, e) for e, t in _SPAN_PAIR_RE.findall(answer_part)]
    return pairs
