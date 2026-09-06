# -*- coding: utf-8 -*-
"""eval_v3: 最终评测脚本(汇总实验结果 + Markov 基线 + 生成质量/和声贴合)。

产出:
  REPORT_TABLE.md   可读的结果表;
  experiments/summary.json   机器可读的全部数字(供 README/HIGHLIGHTS 回填)。

用法:
  python eval_v3.py --base base_s1 --cond cond_s1 [--xformer xf_s1] [--num-seeds 12]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import corpus as C
import generate_v3 as G

PROJECT_ROOT = Path(__file__).resolve().parent
EXP_DIR = PROJECT_ROOT / "experiments"
OUT_TABLE = PROJECT_ROOT / "REPORT_TABLE.md"
OUT_SUM = EXP_DIR / "summary.json"
PROG = ["C", "Am", "F", "G"]
STEPS_PER_BAR = 16
SEED_LEN = 24
RANDOM_PC_BASELINE = 3.0 / 12.0  # 任意一个音高落在三和弦(3/12)内的概率


def model_rows():
    """扫描 experiments/*_results.json, 返回全部已跑完的模型指标(含困惑度)。"""
    rows = []
    for p in sorted(EXP_DIR.glob("*_results.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        r["ppl"] = math.exp(r["test_loss"])
        rows.append(r)
    return rows


def markov_test(corpus, n_test=4000):
    """6-gram Markov 基线: 只拟合 train split 的 token, 在固定 test 窗口上报 loss/acc。"""
    train_tok, _ = corpus.split_sequences("train")
    x, _, y = corpus.fixed_windows("test", n_test)
    mb = C.MarkovBaseline(order=6)
    mb.fit(train_tok)
    nll_sum = acc_hit = n = 0
    for i in range(len(x)):
        probs, mass = mb._prob(x[i], corpus.n_sym)
        p = probs.get(int(y[i]), 1e-6 / corpus.n_sym)
        nll_sum += -math.log(max(p * mass, 1e-12))
        if probs and max(probs, key=probs.get) == int(y[i]):
            acc_hit += 1
        n += 1
    return nll_sum / n, acc_hit / n


def pick_seeds(corpus, n=12, melody_len=SEED_LEN, seed=7):
    """从 test split(训练从未见过)抽 n 条 prompt。

    与"随机切 24 个符号"不同, 这里要求窗口落在一条主旋律内部、且**以一个新
    音符(而非长音/休止)结尾** —— 模拟"唱到这里, 请继续"的真实续写场景,
    避免 24 个符号里全是长音/休止导致两种模型都立即自然收束。
    """
    rng = np.random.default_rng(seed)
    pool = []
    for i in corpus.splits["test"]:
        toks = (C.V3_DIR / f"{i}_tokens.txt").read_text(encoding="utf-8").split()
        if len(toks) >= melody_len + 16:
            pool.append(toks)
    rng.shuffle(pool)
    seeds = []
    for toks in pool:
        # 候选: 窗口需能容纳 24 个符号、且结尾音符之后至少还剩 16 个符号(约 1 小节)
        cand = []
        note_pos = [j for j, t in enumerate(toks) if t not in ("_", "r")]
        for p in note_pos:
            s = p - melody_len + 1
            if s < 0 or len(toks) - 1 - p < 16:
                continue
            seg = toks[s:p + 1]
            if sum(1 for t in seg if t not in ("_", "r")) >= 4:
                cand.append(seg)
        if cand:
            seeds.append(cand[int(rng.integers(len(cand)))])
        if len(seeds) >= n:
            break
    assert len(seeds) >= n, f"只找到 {len(seeds)}/{n} 条合适 prompt"
    return seeds


def harmonic_fit_corpus(corpus, name="test"):
    """真实主旋律在自身和弦上的贴合率(校准参考: 人声也含大量经过音/倚音)。"""
    fit = tot = 0
    for i in corpus.splits[name]:
        toks = (C.V3_DIR / f"{i}_tokens.txt").read_text(encoding="utf-8").split()
        chds = (C.V3_DIR / f"{i}_chords.txt").read_text(encoding="utf-8").split()
        for t, ch in zip(toks, chds):
            if t in ("_", "r", "/"):
                continue
            tot += 1
            fit += int(int(t) % 12 in G.chord_tones(ch))
    return (fit / tot) if tot else float("nan"), tot


def _note_fit(seq, seeds_len, labels):
    """生成的 note-on 里, 落在所在小节和弦三音内的比例(从新生成的 step 0 对齐)。"""
    fit = tot = 0
    for i, t in enumerate(seq[seeds_len:]):
        if t in ("_", "r", "/"):
            continue
        rel = i
        lab = labels[(rel // STEPS_PER_BAR) % len(labels)]
        tot += 1
        fit += int(int(t) % 12 in G.chord_tones(lab))
    return fit, tot


def pc_dist(tokens_or_ints):
    pc = np.zeros(12)
    for t in tokens_or_ints:
        if t in ("_", "r", "/"):
            continue
        pc[int(t) % 12] += 1
    return pc


def pc_kld(a, b, eps=1e-6):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a / a.sum(); b = b / b.sum()
    return float(np.sum(a * np.log((a + eps) / (b + eps))))


def generation_summary(corpus, model, kind, seeds, labels, num_steps=160, temperature=0.7):
    import keras
    stats, all_onsets, fit_sum, fit_tot = [], [], 0, 0
    for seed in seeds:
        if kind == "base":
            seq, stopped = G.gen_base(model, corpus, seed, num_steps, temperature)
        else:
            psteps = G.progression_steps(labels)
            seq, stopped = G.gen_cond(model, corpus, seed, psteps, num_steps, temperature)
        tail = seq[len(seed):]
        rep = float("nan")
        if len(tail) >= 9:
            grams = [tuple(tail[i:i + 8]) for i in range(len(tail) - 7)]
            rep = 1.0 - len(set(grams)) / len(grams)
        onsets = [int(t) for t in tail if t not in ("_", "r", "/")]
        all_onsets.extend(onsets)
        ent = float("nan")
        if onsets:
            cnt = np.bincount(np.asarray(onsets) % 12, minlength=12)
            p = cnt / cnt.sum()
            ent = float(-np.sum(p * np.log2(p + 1e-12)))
        stats.append({
            "len": len(tail), "stopped": stopped, "rep": rep, "entropy": ent,
            "range": (max(onsets) - min(onsets)) if onsets else 0,
        })
        f, t = _note_fit(seq, len(seed), labels)
        fit_sum += f; fit_tot += t
    pc = pc_dist(all_onsets)
    return {
        "n_seeds": len(seeds),
        "avg_len": float(np.nanmean([s["len"] for s in stats])),
        "stop_rate": float(np.mean([s["stopped"] for s in stats])),
        "rep8": float(np.nanmean([s["rep"] for s in stats])),
        "entropy": float(np.nanmean([s["entropy"] for s in stats])),
        "range": float(np.nanmean([s["range"] for s in stats])),
        "n_notes": int(len(all_onsets)),
        "harmonic_fit": fit_sum / fit_tot if fit_tot else float("nan"),
        "pc": pc.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--num-steps", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    co = C.Corpus()
    rows = model_rows()

    # Markov 基线(只看 token, 与 base 相同输入信息)
    m_nll, m_acc = markov_test(co)
    rows.append({"name": "markov-6", "kind": "markov",
                 "test_loss": m_nll, "test_acc": m_acc, "ppl": math.exp(m_nll)})

    # 真实语料参考(音高类别分布 + 和弦贴合)
    real_pc = pc_dist(t for i in co.splits["train"]
                      for t in (C.V3_DIR / f"{i}_tokens.txt").read_text(encoding="utf-8").split())
    hf_real, hf_n = harmonic_fit_corpus(co, "test")
    real_kld_note = None

    seeds = pick_seeds(co, args.num_seeds)
    labels = PROG
    import keras
    gen_rows = []
    for r in rows:
        name, kind = r["name"], r["kind"]
        if kind not in ("base", "cond"):
            continue
        p = EXP_DIR / f"{name}.keras"
        if not p.exists():
            continue
        model = keras.models.load_model(str(p))
        g = generation_summary(co, model, kind, seeds, labels,
                               num_steps=args.num_steps, temperature=args.temperature)
        g["kld_pc_vs_corpus"] = pc_kld(g["pc"], real_pc)
        g["name"], g["kind"] = name, kind
        gen_rows.append(g)

    summary = {
        "progression": labels,
        "seeds": args.num_seeds,
        "model_metrics": rows,
        "harmonic_fit": {
            "random_baseline": RANDOM_PC_BASELINE,
            "real_melody_on_own_chords": hf_real,
            "n_real_notes": hf_n,
            "generated": [{"name": g["name"], "fit": g["harmonic_fit"]} for g in gen_rows],
        },
        "generation": gen_rows,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    # ---- 可读表格 ----
    order = {"markov-6": 0, "base_s1": 1, "cond_s1": 2,
             "base_r1": 3, "cond_r1": 4, "xf_s1": 5}
    lines = []
    lines.append("# 实验结果表(Final Evaluation)\n")
    lines.append("固定验证/测试窗口对全部模型开放; 生成评测使用同一组 12 条 test 旋律 prompt。\n")
    lines.append("## 1. 逐符号指标(固定 test 窗口, 4,000 窗口)")
    lines.append("")
    lines.append("| 模型 | 类型 | test NLL | test acc | 困惑度 PPL |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: order.get(r["name"], 9)):
        lines.append(f"| {r['name']} | {r['kind']} | {r['test_loss']:.4f} | "
                     f"{r['test_acc']:.4f} | {r['ppl']:.3f} |")
    lines.append("")
    lines.append("## 2. 和弦贴合(生成 vs C-Am-F-G, note-on 落在所在小节和弦三音内的比例)")
    lines.append("")
    lines.append("| 模型 | 和弦贴合 | 参考: 随机音高 | 参考: 真实人声主旋律 |")
    lines.append("|---|---|---|---|")
    for g in gen_rows:
        lines.append(f"| {g['name']} ({g['kind']}) | {g['harmonic_fit'] * 100:.1f}% | "
                     f"{RANDOM_PC_BASELINE * 100:.1f}% | {hf_real * 100:.1f}% |")
    lines.append("")
    lines.append("## 3. 生成质量(固定 12 条 test prompt、C-Am-F-G)")
    lines.append("")
    lines.append("| 模型 | 平均长度 | 音符数(合计) | 自然停笔率 | 8-gram 重复率 | 音高熵(bit) | 音域 | 音高类 KLD |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for g in gen_rows:
        lines.append(f"| {g['name']} ({g['kind']}) | {g['avg_len']:.0f} | {g['n_notes']} | "
                     f"{g['stop_rate']:.2f} | {g['rep8']:.3f} | {g['entropy']:.2f} | "
                     f"{g['range']:.1f} | {g['kld_pc_vs_corpus']:.3f} |")
    lines.append("")
    (OUT_TABLE).write_text("\n".join(lines), encoding="utf-8")
    print(OUT_TABLE, "written")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()