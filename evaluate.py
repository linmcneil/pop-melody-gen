"""统一评测工具,回答“到底改进了多少”。

用法一:看模型预测能力(验证集 loss / accuracy),v1 与 v2 用同一段 v1 没见过的窗口对比。
  python evaluate.py metrics --model ../melody-rnn-lstm/model.keras --kind onehot
  python evaluate.py metrics --model model.keras --kind int

用法二:看生成质量。固定同一批种子、同一个模型,只切换采样器,
  对比 naive(温度采样,对应 v1 做法)与 v2(温度 + top-p + 重复惩罚)的输出。
  python evaluate.py compare --model model.keras --kind int

指标说明(生成侧):
  stop_rate    在 num_steps 内模型自己采到 "/" 自然结束的比例;
  mean_len     平均续写的符号数(自然结束的曲子长短);
  rep_index    尾部 8-gram 重复占比(越低越不“复读机”);
  max_run      最长连续相同符号(越低越不容易卡在一个音上);
  onset_num    真正落下的新音符个数;
  pc_entropy   音高类别(音名)分布的熵(越高越多样);
  pitch_range  音高跨度(MIDI 差,越大越有起伏)。
"""

import argparse
import collections
import math
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import generate as gen
from preprocess import SEQUENCE_LENGTH, SINGLE_FILE_DATASET, build_windows, load_mapping

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model.keras"


# ---------------- metrics:验证集 loss / accuracy ----------------

def make_val_dataset(x_int, y_int, kind, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((x_int, y_int))
    if kind == "onehot":
        vocab = int(np.max(x_int)) + 1
        ds = ds.map(lambda xb, yb: (tf.one_hot(xb, depth=vocab), yb))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def run_metrics(args):
    mapping = load_mapping()
    vocab_size = len(mapping)
    total = args.train_samples + args.val_samples
    x_all, y_all = build_windows(SEQUENCE_LENGTH, max_samples=total)
    x_val, y_val = x_all[args.train_samples:], y_all[args.train_samples:]
    print(f"验证窗口数: {len(x_val)}(窗口 {args.train_samples}~{total})")

    model = keras.models.load_model(args.model)
    val_ds = make_val_dataset(x_val, y_val, args.kind, args.batch_size)
    result = model.evaluate(val_ds, verbose=1)
    loss, acc = result[0], result[1]
    print("\n===== 模型指标 =====")
    print(f"模型: {args.model}")
    print(f"验证 loss: {loss:.4f} | 验证 accuracy: {acc:.4f}")
    print(f"(词表 {vocab_size};随机猜测准确率约 {100.0 / vocab_size:.2f}%)")


# ---------------- compare:生成质量对比 ----------------

def _pc_entropy(onsets):
    if not onsets:
        return 0.0
    counter = collections.Counter(n % 12 for n in onsets)
    total = sum(counter.values())
    return -sum(
        (c / total) * math.log2(c / total) for c in counter.values()
    )


def _tail_metrics(tail):
    """tail 是模型在种子之后续写出来的符号序列。"""
    rep_index = 0.0
    if len(tail) >= 9:
        grams = [tuple(tail[i:i+8]) for i in range(len(tail) - 7)]
        rep_index = 1.0 - len(set(grams)) / len(grams)

    max_run = 0
    run = 0
    prev = None
    for token in tail:
        if token == prev:
            run += 1
        else:
            run = 1
            prev = token
        max_run = max(max_run, run)

    onsets = [int(t) for t in tail if t not in ("_", "r", "/")]
    return {
        "rep_index": rep_index,
        "max_run": max_run,
        "onset_num": len(onsets),
        "pc_entropy": _pc_entropy(onsets),
        "pitch_range": (max(onsets) - min(onsets)) if onsets else 0.0,
    }


def _collect_results(model, mapping, symbols, seeds, args, sampler):
    rows = []
    for seed_symbols in seeds:
        melody, stopped = gen.generate_melody(
            model, mapping, symbols, seed_symbols,
            kind=args.kind,
            sampler=sampler,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            rep_window=args.rep_window,
        )
        tail = melody[len(seed_symbols):]
        metrics = _tail_metrics(tail)
        rows.append({
            "stopped": stopped,
            "len": len(tail),
            **metrics,
        })
    return rows


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def _fmt_pct(a, b):
    """b 相对 a 的变化百分比。"""
    if abs(a) < 1e-9:
        return "n/a"
    return f"{100.0 * (b - a) / a:+.1f}%"


def run_compare(args):
    model, mapping, symbols, _ = gen.load_model_and_mapping(args.model)

    # 从语料里随机抽 num_seeds 个不含 "/" 的 16 步片段当种子
    tokens = SINGLE_FILE_DATASET.read_text(encoding="utf-8").split()
    rng = np.random.RandomState(args.seed)
    seeds = []
    attempts = 0
    while len(seeds) < args.num_seeds and attempts < 100000:
        attempts += 1
        start = rng.randint(0, len(tokens) - 16)
        piece = tokens[start:start + 16]
        if "/" in piece:
            continue
        seeds.append(piece)
    if len(seeds) < args.num_seeds:
        raise RuntimeError("语料里没抽够不含分隔符的种子片段")

    print(f"随机种子片段数: {len(seeds)},续写步数上限: {args.num_steps}")

    naive_rows = _collect_results(model, mapping, symbols, seeds, args, "naive")
    v2_rows = _collect_results(model, mapping, symbols, seeds, args, "v2")

    headers = ["指标", "naive", "v2", "相对变化"]
    rows = [
        ["自然结束率 stop_rate",
         f"{_mean(naive_rows, 'stopped'):.2f}",
         f"{_mean(v2_rows, 'stopped'):.2f}",
         _fmt_pct(_mean(naive_rows, "stopped"), _mean(v2_rows, "stopped"))],
        ["平均续写长度 mean_len",
         f"{_mean(naive_rows, 'len'):.1f}",
         f"{_mean(v2_rows, 'len'):.1f}",
         _fmt_pct(_mean(naive_rows, "len"), _mean(v2_rows, "len"))],
        ["8-gram 重复率 rep_index",
         f"{_mean(naive_rows, 'rep_index'):.4f}",
         f"{_mean(v2_rows, 'rep_index'):.4f}",
         _fmt_pct(_mean(naive_rows, "rep_index"), _mean(v2_rows, "rep_index"))],
        ["最长重复串 max_run",
         f"{_mean(naive_rows, 'max_run'):.1f}",
         f"{_mean(v2_rows, 'max_run'):.1f}",
         _fmt_pct(_mean(naive_rows, "max_run"), _mean(v2_rows, "max_run"))],
        ["新音符数 onset_num",
         f"{_mean(naive_rows, 'onset_num'):.1f}",
         f"{_mean(v2_rows, 'onset_num'):.1f}",
         _fmt_pct(_mean(naive_rows, "onset_num"), _mean(v2_rows, "onset_num"))],
        ["音高类别熵 pc_entropy",
         f"{_mean(naive_rows, 'pc_entropy'):.3f}",
         f"{_mean(v2_rows, 'pc_entropy'):.3f}",
         _fmt_pct(_mean(naive_rows, "pc_entropy"), _mean(v2_rows, "pc_entropy"))],
        ["音高跨度 pitch_range",
         f"{_mean(naive_rows, 'pitch_range'):.1f}",
         f"{_mean(v2_rows, 'pitch_range'):.1f}",
         _fmt_pct(_mean(naive_rows, "pitch_range"), _mean(v2_rows, "pitch_range"))],
    ]

    print("\n===== 生成质量对比(同一模型,仅换采样器)=====")
    widths = [max(len(h) for h in headers) + 2] + [14, 14, 14]
    for i, header in enumerate(headers):
        print(header.ljust(widths[i]), end="")
    print()
    for row in rows:
        for i, cell in enumerate(row):
            print(str(cell).ljust(widths[i]), end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="v2 评测")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_m = sub.add_parser("metrics", help="验证集 loss/accuracy")
    p_m.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p_m.add_argument("--kind", choices=["int", "onehot"], default="int")
    p_m.add_argument("--train-samples", type=int, default=20000)
    p_m.add_argument("--val-samples", type=int, default=4000)
    p_m.add_argument("--batch-size", type=int, default=128)
    p_m.set_defaults(func=run_metrics)

    p_c = sub.add_parser("compare", help="生成质量对比")
    p_c.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p_c.add_argument("--kind", choices=["int", "onehot"], default="int")
    p_c.add_argument("--num-seeds", type=int, default=24)
    p_c.add_argument("--num-steps", type=int, default=300)
    p_c.add_argument("--temperature", type=float, default=0.4)
    p_c.add_argument("--top-p", type=float, default=0.92)
    p_c.add_argument("--repetition-penalty", type=float, default=0.5)
    p_c.add_argument("--rep-window", type=int, default=8)
    p_c.add_argument("--seed", type=int, default=7)
    p_c.set_defaults(func=run_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()