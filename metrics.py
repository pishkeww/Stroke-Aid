import math

import numpy as np

NAN = float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (NAN, NAN)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)


def confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    return tp, fp, tn, fn


def _ratio(k, n):
    return k / n if n else NAN


def diag_stats(y_true, y_pred):
    tp, fp, tn, fn = confusion(y_true, y_pred)
    return {
        "n": tp + fp + tn + fn, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": _ratio(tp, tp + fn), "sens_ci": wilson(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp), "spec_ci": wilson(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp), "npv": _ratio(tn, tn + fn),
    }


def auc(y_true, score):
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return NAN
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    vals = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(vals))
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    rank_of = np.empty(len(vals))
    rank_of[order] = ranks
    r_pos = rank_of[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_auc_ci(y_true, score, n_boot=1000, seed=0):
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        a = auc(y[idx], s[idx])
        if not math.isnan(a):
            vals.append(a)
    if len(vals) < 20:
        return (NAN, NAN)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def sweep(y_true, score, thresholds):
    out = []
    for t in thresholds:
        d = diag_stats(y_true, [s >= t for s in score])
        d["threshold"] = t
        out.append(d)
    return out
