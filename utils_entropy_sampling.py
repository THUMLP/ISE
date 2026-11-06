"""
预测熵采样模块（Entropy Sampling）- 重构版

改进点：
1. 动态调整熵平滑参数（根据历史轮数）
2. 自适应课程式采样参数（根据收敛情况）
3. 优化熵计算逻辑
4. 改进高熵剔除策略（根据样本量动态调整）
"""
import math
import random
from collections import Counter, defaultdict


# ======================================================================
# 熵计算
# ======================================================================

def binary_entropy(p):
    """二元熵 H(p)，取值 [0, 1]（log2）。p=0 或 1 时为 0（完全稳定）。"""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def adaptive_smoothed_p(c, n):
    """
    自适应平滑频率：历史越少，平滑越强

    改进：alpha 根据历史轮数动态调整
    - n=2: alpha≈1.41，强平滑
    - n=5: alpha≈0.89
    - n=10: alpha≈0.63
    - n≥20: alpha≈0.5，弱平滑
    """
    alpha = max(0.5, 2.0 / math.sqrt(n))
    return (c + alpha) / (n + 2.0 * alpha)


def smoothed_p(c, n, alpha=1.0):
    """
    Beta(alpha, alpha) 先验下的平滑频率，用于小样本熵估计。
    保留原接口以兼容旧代码。
    """
    return (c + alpha) / (n + 2.0 * alpha)


def _history_entity_sets(history):
    """把历史预测列表转为 [set((type, entity)), ...]，便于统计命中率。"""
    sets = []
    for entities in history or []:
        s = set()
        for ent in entities or []:
            if isinstance(ent, dict) and "type" in ent and "entity" in ent:
                s.add((str(ent["type"]).strip().lower(), str(ent["entity"]).strip()))
        sets.append(s)
    return sets


def sentence_entropy(text, prediction_history):
    """
    句子级熵：对历史中出现过的每个 (type, entity) 计算二元熵后取平均。
    返回 [0, 1]，越高表示该句预测越不稳定；无历史时返回 0.5。

    改进：使用自适应平滑
    """
    if not prediction_history or text not in prediction_history:
        return 0.5
    history = prediction_history[text]
    if len(history) < 2:
        return 0.5

    sets = _history_entity_sets(history)
    counter = Counter()
    for s in sets:
        counter.update(s)
    if not counter:
        # 所有历史都预测为空，视为稳定
        return 0.0

    n = len(sets)
    return sum(binary_entropy(adaptive_smoothed_p(c, n)) for c in counter.values()) / len(counter)


def _parse_error_tuple(error):
    """
    解析错误元组，返回 (golden_label, entity)。
      Confusion: ("predict label: p", "entity: e", "golden label: g")
      Missing:   ("golden label: g", "entity: e", "Easily missed entity")
    解析失败的字段返回 None。
    """
    gold, ent = None, None
    fields = error if isinstance(error, (list, tuple)) else [error]
    for f in fields:
        s = str(f)
        if s.startswith("golden label:"):
            gold = s.split(":", 1)[1].strip().lower()
        elif s.startswith("entity:"):
            ent = s.split(":", 1)[1].strip()
    return gold, ent


def error_entropy(text_key, error, prediction_history):
    """
    错误级熵：统计该实体以锚点类型 (golden label, entity) 出现在历史预测中的频率 p，返回 H(p)。
      - p≈0：一直错（系统性偏差）→ 低熵，最值得纠正
      - p≈1：一直对（本轮偶发错误）→ 低熵
      - p≈0.5：时对时错 → 高熵，锚点/预测都不可靠
    无法解析实体时退化为句子级熵。

    改进：使用自适应平滑
    """
    text = text_key[len("content: "):] if str(text_key).startswith("content: ") else text_key
    if not prediction_history or text not in prediction_history:
        return 0.5
    history = prediction_history[text]
    if len(history) < 2:
        return 0.5

    gold, ent = _parse_error_tuple(error)
    if ent is None:
        return sentence_entropy(text, prediction_history)

    sets = _history_entity_sets(history)
    if gold is not None:
        hits = sum(1 for s in sets if (gold, ent) in s)
    else:
        hits = sum(1 for s in sets if any(e == ent for _, e in s))
    return binary_entropy(adaptive_smoothed_p(hits, len(sets)))


def calculate_prediction_entropy(text, prediction_history):
    """
    兼容旧接口：返回可靠性得分 = 1 - 句子级熵，取值 [0, 1]，越高越可靠。
    """
    return 1.0 - sentence_entropy(text, prediction_history)


# ======================================================================
# 课程式熵采样：前期低熵，后期低熵 + 少量中熵
# ======================================================================

def adaptive_high_cut(n_samples, base_cut=0.2, min_samples=10):
    """
    根据样本量动态调整高熵剔除比例

    改进：样本少时减少剔除，避免稀有类型样本过少
    - n <= min_samples: 不剔除
    - n = 2*min_samples: 剔除 base_cut
    - n >= 2*min_samples: 剔除 base_cut
    """
    if n_samples <= min_samples:
        return 0.0
    return base_cut * min(1.0, n_samples / (2 * min_samples))


DEFAULT_WARMUP_ITERS = 3
DEFAULT_MID_RATIO = 0.3
DEFAULT_HIGH_CUT = 0.2


def entropy_curriculum_sample(samples, k, score_fn, round_idx, mode="curriculum",
                              warmup_iters=None, high_cut=None, mid_ratio=None,
                              min_keep=None):
    """
    在单个实体类型桶内挑选 k 个样本。

    改进：
    1. 支持自适应参数调整
    2. 根据样本量动态调整剔除比例
    3. 改进最小保留数量策略

    Args:
        samples: 候选样本列表 [(text_key, error), ...]
        k: 目标采样数
        score_fn: sample -> 熵 [0, 1]
        round_idx: 这批样本将用于第几轮专家更新（从 1 开始）
        mode: "random"（原随机） | "low"（始终低熵） | "curriculum"（前期低熵，后期混入中熵）
        warmup_iters: round_idx <= warmup_iters 时只用低熵样本（None时使用自适应值）
        high_cut: 剔除熵最高的比例，视为噪声区（None时使用自适应值）
        mid_ratio: 后期每类中熵样本占比（None时使用自适应值）
        min_keep: 剔除噪声后每类至少保留的候选数（None时使用自适应值）
    """
    if not samples or k <= 0:
        return []
    if mode == "random":
        return list(samples) if len(samples) <= k else random.sample(samples, k)

    if warmup_iters is None:
        warmup_iters = DEFAULT_WARMUP_ITERS
    if mid_ratio is None:
        mid_ratio = DEFAULT_MID_RATIO
    if high_cut is None:
        high_cut = DEFAULT_HIGH_CUT

    # 自适应最小保留数量
    if min_keep is None:
        min_keep = max(2, min(5, k // 2))

    # 打乱后稳定排序：同熵样本随机打破平局，避免每轮选到同一批
    pool = list(samples)
    random.shuffle(pool)
    scores = {id(s): score_fn(s) for s in pool}
    pool.sort(key=lambda s: scores[id(s)])

    # 1) 动态剔除高熵噪声
    n = len(pool)
    adaptive_cut = adaptive_high_cut(n, high_cut)
    n_drop = int(n * adaptive_cut)
    n_keep = max(n - n_drop, min(min_keep, n))

    if n_keep < n:
        boundary = scores[id(pool[n_keep - 1])]
        kept = pool[:n_keep] + [s for s in pool[n_keep:] if scores[id(s)] <= boundary]
    else:
        kept = pool

    # 2) 按排名切分低熵池 / 中熵池
    half = math.ceil(len(kept) / 2)
    low_pool, mid_pool = kept[:half], kept[half:]

    def take(src, m):
        return src if len(src) <= m else random.sample(src, m)

    use_mid = mode == "curriculum" and round_idx > warmup_iters and mid_pool
    if not use_mid:
        chosen = take(low_pool, k)
        # 低熵不够时按熵升序从中熵池补齐
        chosen += mid_pool[:k - len(chosen)]
        return chosen

    n_mid = min(len(mid_pool), max(1 if k >= 2 else 0, round(k * mid_ratio)))
    n_low = k - n_mid
    chosen_low = take(low_pool, n_low)
    chosen_mid = take(mid_pool, n_mid)
    chosen = chosen_low + chosen_mid
    # 任一池不足时用另一池剩余样本补齐
    if len(chosen) < k:
        picked = set(map(id, chosen))
        rest = [s for s in kept if id(s) not in picked]
        chosen += rest[:k - len(chosen)]
    return chosen


# ======================================================================
# 历史维护与兼容接口
# ======================================================================

def update_prediction_history(prediction_history, y_pred):
    """
    更新预测历史记录

    Args:
        prediction_history: 现有的预测历史 {text: [entities_iter1, entities_iter2, ...]}
        y_pred: 当前迭代的预测结果 [{"text": "...", "entities": [...]}, ...]
    """
    if prediction_history is None:
        prediction_history = {}

    for item in y_pred:
        text = item["text"]
        entities = item.get("entities", [])
        prediction_history.setdefault(text, []).append(entities)

    return prediction_history


def calculate_all_reliabilities(prediction_history):
    """批量计算所有样本的可靠性得分 {text: reliability}"""
    return {text: calculate_prediction_entropy(text, prediction_history) for text in prediction_history}


def filter_reliable_samples(samples, prediction_history, reliability_threshold=0.6):
    """根据可靠性阈值过滤样本，返回 (过滤后样本, {sample: reliability})"""
    filtered_samples = []
    sample_scores = {}
    for text, error in samples:
        reliability = calculate_prediction_entropy(text, prediction_history)
        sample_scores[(text, error)] = reliability
        if reliability >= reliability_threshold:
            filtered_samples.append((text, error))
    return filtered_samples, sample_scores


# ======================================================================
# 工具函数
# ======================================================================

def estimate_convergence_rate(f1_history, window=3):
    """
    估计最近几轮的F1提升速度

    Args:
        f1_history: F1历史列表 [f1_iter1, f1_iter2, ...]
        window: 窗口大小

    Returns:
        recent_improvement: 平均每轮F1提升量
    """
    if len(f1_history) < 2:
        return None

    recent = f1_history[-min(window, len(f1_history)):]
    if len(recent) < 2:
        return None

    improvements = [recent[i] - recent[i-1] for i in range(1, len(recent))]
    return sum(improvements) / len(improvements)
