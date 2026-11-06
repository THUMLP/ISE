"""
伪标签锚点精炼模块（Anchor Refinement）

在不增加 API 调用的前提下提升伪标签集合质量：
    1. 同一 span 只保留一个类型：按 (当前轮专家票数 + 融合权重) 打分，平票依次取融合结果、旧锚点
    2. 有效性过滤：span 必须是原文子串（评估为精确匹配，否则必为 FP）；类型必须属于 ENTITY_TYPES
    3. 第 0 轮锚点：多次融合多数投票，首个融合（temperature=0）作为类型平票裁决
    4. 跨轮历史集成：利用 prediction_history 中全部历史专家预测
         - (type, entity) 历史频率 >= hist_accept   -> 直接收（找回被一次性丢掉的稳定实体）
         - span（不区分类型）历史频率 < hist_reject -> 直接拒（专家几乎从不认为它是实体）
         - 其余实体沿用原规则 C^(t) ∪ (E_int^(t) ∩ S_pseudo^(t-1))
       span 级拒绝而非类型级拒绝：融合把 person 改判为 researcher 这类纠正不会因专家历史类型不同而被误删
"""
from collections import Counter, defaultdict


def _key(ent):
    """实体 -> (type 小写, entity)；非法结构返回 None。"""
    if isinstance(ent, dict) and "type" in ent and "entity" in ent:
        t, e = str(ent["type"]).strip().lower(), str(ent["entity"]).strip()
        if t and e:
            return t, e
    return None


def _index_by_text(preds):
    """[{"text", "entities"}] -> {text: [entities]}，按文本建索引，避免顺序错位。"""
    d = defaultdict(list)
    for item in preds or []:
        if isinstance(item, dict) and "text" in item:
            d[item["text"]].extend(item.get("entities") or [])
    return d


def _key_map(entities):
    """[entities] -> {key: 首次出现的原始实体对象}"""
    m = {}
    for ent in entities:
        k = _key(ent)
        if k is not None and k not in m:
            m[k] = ent
    return m


def _ordered_texts(*dicts):
    return list(dict.fromkeys(t for d in dicts for t in d))


def is_valid_entity(key, text, target_types):
    t, e = key
    return t in target_types and e in text


def history_frequencies(history):
    """返回 (历史预测次数, 类型级命中 Counter, span 级命中 Counter)，每次预测内去重计数。"""
    typed, spans = Counter(), Counter()
    for ents in history:
        keys = {k for k in map(_key, ents or []) if k is not None}
        typed.update(keys)
        spans.update({e for _, e in keys})
    return len(history), typed, spans


def resolve_span_types(keys, score_fn):
    """同一 span 存在多个类型时只保留得分最高者。返回 (保留集合, 冲突 span 数)。"""
    by_span = defaultdict(list)
    for k in keys:
        by_span[k[1]].append(k)
    kept, n_conflict = set(), 0
    for cands in by_span.values():
        if len(cands) > 1:
            n_conflict += 1
        kept.add(max(cands, key=score_fn))
    return kept, n_conflict


def _target_types(entity_types):
    return {str(t).strip().lower() for t in entity_types}


def init_anchor_from_fusion_runs(fusion_runs, entity_types, threshold=None, logger=print):
    """
    第 0 轮锚点：K 次融合结果多数投票（>= threshold，默认 K//2+1）。
    fusion_runs[0] 应为确定性融合（temperature=0），用于同 span 类型平票裁决。
    与原实现一致，只返回至少含一个实体的文本。
    """
    k_runs = len(fusion_runs)
    threshold = threshold or (k_runs // 2 + 1)
    target_types = _target_types(entity_types)
    run_dicts = [_index_by_text(r) for r in fusion_runs]
    stats = Counter()

    anchor = []
    for text in _ordered_texts(*run_dicts):
        maps = [_key_map(d.get(text, [])) for d in run_dicts]
        votes = Counter(k for m in maps for k in m)
        cands = {k for k, v in votes.items() if v >= threshold}

        invalid = {k for k in cands if not is_valid_entity(k, text, target_types)}
        stats["invalid"] += len(invalid)
        cands -= invalid

        kept, n_conf = resolve_span_types(cands, lambda k: (votes[k], k in maps[0], k[0]))
        stats["type_conflict"] += n_conf

        ents = [next(m[k] for m in maps if k in m) for k in sorted(kept)]
        if ents:
            anchor.append({"text": text, "entities": ents})

    logger(f"   >>> [锚点精炼-初始化] 投票阈值 {threshold}/{k_runs}，过滤无效 {stats['invalid']}，"
           f"类型冲突消解 {stats['type_conflict']}，锚点实体数 {sum(len(a['entities']) for a in anchor)}")
    return anchor


def conservative_anchor_update(prev_anchor, new_merged, expert_preds_list, entity_types,
                               prediction_history=None, consensus_threshold=2,
                               fusion_type_weight=1.0, fusion_vote_weight=2.0,
                               hist_accept=0.7, hist_reject=0.0,
                               hist_min_len=6, logger=print):
    """
    保守锚点更新 + 精炼：.
        S^(t) = Resolve( Valid( [C^(t) ∪ (E_int^(t) ∩ S^(t-1)) ∪ H_acc] \\ H_rej ) )

    Args:
        prev_anchor:        S^(t-1)
        new_merged:         E_int^(t)，本轮融合结果
        expert_preds_list:  本轮各专家预测
        entity_types:       合法类型集合
        prediction_history: {text: [历史各次专家预测的实体列表]}；为空则跳过历史规则
        fusion_type_weight: 类型冲突打分时融合结果折算的票数（0 = 只看专家票数）
        fusion_vote_weight: 融合结果计入 consensus 投票的票数权重（默认 0，不改变现有行为，
                             仅按旧的交集规则兜底；传入 >0 比如 1.0 才会把融合结果当第四个
                             投票源，与一个专家等权参与 consensus_threshold 门槛）
        hist_accept:        类型级历史频率 >= 该值直接收；> 1 关闭
        hist_reject:        span 级历史频率 < 该值直接拒；<= 0 关闭
        hist_min_len:       历史预测次数不足时不启用历史规则
    """
    target_types = _target_types(entity_types)
    expert_dicts = [_index_by_text(p) for p in expert_preds_list]
    fusion_d = _index_by_text(new_merged)
    prev_d = _index_by_text(prev_anchor)
    prediction_history = prediction_history or {}
    stats = Counter()

    new_anchor = []
    for text in _ordered_texts(prev_d, fusion_d):
        votes, expert_objs = Counter(), {}
        for ed in expert_dicts:
            km = _key_map(ed.get(text, []))
            votes.update(km.keys())
            for k, ent in km.items():
                expert_objs.setdefault(k, ent)
        fusion_m = _key_map(fusion_d.get(text, []))
        prev_m = _key_map(prev_d.get(text, []))

        # 融合结果作为独立投票源参与 consensus 门槛（权重 fusion_vote_weight），
        # 使其能够贡献三专家都未发现、但融合综合三份定义后识别出的实体。
        # 单独开一个 consensus_votes，避免与下面类型裁决里的 fusion_type_weight 重复计数。
        consensus_votes = Counter(votes)
        if fusion_vote_weight:
            for k in fusion_m:
                consensus_votes[k] += fusion_vote_weight

        # 规则：C^(t) ∪ (E_int^(t) ∩ S^(t-1))
        cands = {k for k, v in consensus_votes.items() if v >= consensus_threshold}
        cands |= fusion_m.keys() & prev_m.keys()

        # 4. 跨轮历史集成
        typed = Counter()
        history = prediction_history.get(text) or []
        if len(history) >= hist_min_len:
            n, typed, spans = history_frequencies(history)
            accepted = {k for k, c in typed.items() if c / n >= hist_accept} - cands
            stats["hist_accept"] += len(accepted)
            cands |= accepted
            rejected = {k for k in cands if spans[k[1]] / n < hist_reject}
            stats["hist_reject"] += len(rejected)
            cands -= rejected

        # 2. 有效性过滤
        invalid = {k for k in cands if not is_valid_entity(k, text, target_types)}
        stats["invalid"] += len(invalid)
        cands -= invalid

        # 1. 同 span 单类型：专家票数 + 融合权重；平票依次看融合、旧锚点、历史频率
        kept, n_conf = resolve_span_types(cands, lambda k: (
            votes.get(k, 0) + fusion_type_weight * (k in fusion_m),
            k in fusion_m, k in prev_m, typed.get(k, 0), k[0]))
        stats["type_conflict"] += n_conf

        ents = []
        for k in sorted(kept):
            ent = fusion_m.get(k) or prev_m.get(k) or expert_objs.get(k) or {"type": k[0], "entity": k[1]}
            ents.append(ent)
        new_anchor.append({"text": text, "entities": ents})

    logger(f"   >>> [锚点精炼] 历史收 {stats['hist_accept']}，历史拒 {stats['hist_reject']}，"
           f"过滤无效 {stats['invalid']}，类型冲突消解 {stats['type_conflict']}，"
           f"锚点实体数 {sum(len(a['entities']) for a in new_anchor)}")
    return new_anchor
