# -*- coding: utf-8 -*-
"""human_eval/analysis.py — HUMAN_EVAL.md 协议的分析脚本(纯 numpy, 无需 scipy)。

输入 CSV(UTF-8, 表头):
  rater,loop,A_mel,B_mel,C_mel,A_harm,B_harm,C_harm,A_nat,B_nat,C_nat,A_nov,B_nov,C_nov,choice
  choice = 'A'|'B'|'C' (forced choice: 更像人写的流行主旋律)
对每个评分维度分别跑:
  - 各条件 mean/std;
  - Friedman test(每个 rater×loop 为区组)及近似 p;
  - B vs C 的 Wilcoxon signed-rank(近似 z)与 Bonferroni 校正;
  - 匹配秩双列效应量(Matched rank-biserial)。

用法:
  python human_eval/analysis.py --csv human_eval/results.csv
"""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

DIMS = [("mel", "旋律感"), ("harm", "和声贴合"), ("nat", "自然度"), ("nov", "新颖度")]


def _chi2_sf(x, k):
    """chi-square survival function (自由度 k), 正则化上不完全 gamma 级数。"""
    if x <= 0:
        return 1.0
    a = k / 2.0
    s = t = math.exp(-x / 2.0)
    for i in range(1, 2000):
        t *= x / 2.0 / (a + i - 1)
        s += t
        if t < 1e-12 * s:
            break
    return min(s, 1.0)


def _normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _friedman(blocks):
    """blocks: (n 区组, k 条件) 评分; 返回 Q, p, 列秩和, 有效区组数。"""
    k = blocks.shape[1]
    ranks = np.argsort(np.argsort(blocks, axis=1), axis=1) + 1.0
    n = blocks.shape[0]
    eff_n = n - int((np.max(blocks, axis=1) == np.min(blocks, axis=1)).sum())
    if eff_n < 3:
        return float("nan"), float("nan"), None, eff_n
    R = ranks.sum(axis=0)
    Q = 12.0 / (n * k * (k + 1)) * np.sum(R ** 2) - 3 * n * (k + 1)
    return float(Q), float(_chi2_sf(Q, k - 1)), R, eff_n


def _wilcoxon(a, b):
    """B vs C 的 signed-rank(允许同秩); 返回 z, 双侧 p, n(非零差分数)。"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n < 5:
        return float("nan"), float("nan"), n
    order = np.argsort(np.abs(d))
    d_sorted = d[order]
    abs_sorted = np.abs(d_sorted)
    r = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_sorted[j + 1] == abs_sorted[i]:
            j += 1
        r[i:j + 1] = (i + 1 + j + 1) / 2.0          # 同秩取平均
        i = j + 1
    W = float(np.sum(r[d_sorted > 0]))
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sigma
    p = 2 * (1 - _normal_cdf(abs(z)))
    npos = int(np.sum(d_sorted > 0))
    nneg = int(np.sum(d_sorted < 0))
    rb = 2 * npos / (npos + nneg) - 1 if (npos + nneg) else 0.0
    return float(z), float(p), rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parent / "results.csv"))
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("空 CSV: 还没有可分析的数据。")
        return
    n_rater = len({r["rater"] for r in rows})
    loops = sorted({r["loop"] for r in rows})
    print(f"被试数={n_rater}  loop 数={len(loops)}  评分块={len(rows)}")
    print()

    for suf, name in DIMS:
        vals = {c: [float(r[f"{c}_{suf}"]) for r in rows] for c in "ABC"}
        blocks = np.asarray([[float(r[f"{c}_{suf}"]) for c in "ABC"] for r in rows])
        Q, p, _, eff_n = _friedman(blocks)
        z, pw, rb = _wilcoxon(vals["B"], vals["C"])
        mu = {c: float(np.mean(vals[c])) for c in "ABC"}
        sd = {c: float(np.std(vals[c], ddof=1)) for c in "ABC"}
        print(f"[{name}]  A={mu['A']:.2f}±{sd['A']:.2f}  "
              f"B={mu['B']:.2f}±{sd['B']:.2f}  "
              f"C={mu['C']:.2f}±{sd['C']:.2f}")
        print(f"    Friedman Q={Q:.2f}, p={p:.3g} (df=2, eff.blocks={eff_n})")
        print(f"    Wilcoxon B vs C: z={z:.2f}, p={min(pw * 3, 1):.3g} "
              f"(×3 Bonferroni), matched rank-biserial={rb:+.2f}")

    ch = Counter(r["choice"] for r in rows)
    total = sum(ch.values())
    if total:
        print()
        print("forced choice (更像人写的流行主旋律):")
        for c in "ABC":
            print(f"  {c}: {ch[c]} ({100 * ch[c] / total:.0f}%)")


if __name__ == "__main__":
    main()