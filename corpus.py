# -*- coding: utf-8 -*-
"""语料与窗口: 按歌(melody)划分 train/val/test, 窗口可跨曲但在同一 split 内。

v3 数据管线原则(修复 v1/v2 的"连续前缀切分只用了语料开头一小段"问题):
  1. 以 dataset_v3/<i>_tokens.txt(旋律)+ <i>_chords.txt(逐 token 和弦)为源;
  2. 7000 条旋律按固定随机种子分成 train/val/test(比例 8:1:1), 音乐不跨 split;
  3. 每个 split 内部把旋律拼成一条长序列, 曲间用 '/'*SEP_REPS 与和弦 'N' 填充;
  4. 训练窗口每 epoch 随机抽取(覆盖整个 train split), 验证/测试窗口固定缓存,
     所有模型共用同一批窗口 -> 指标可公平对比。
"""
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
V3_DIR = PROJECT_ROOT / "dataset_v3"
CACHE_DIR = PROJECT_ROOT / "exp_cache"

SEQ_LEN = 64
SEP_REPS = 8
NEUTRAL_CHORD = "N"


class Corpus:
    def __init__(self, split_seed=2025):
        self.split_seed = split_seed
        meta = json.loads((V3_DIR / "splits.json").read_text(encoding="utf-8"))
        self.splits = {
            name: [int(x) for x in (V3_DIR / f"split_{name}.txt").read_text(encoding="utf-8").split()]
            for name in ("train", "val", "test")
        }
        self.symbol_vocab, self.chord_vocab = self._build_vocab()
        self.sym_id = {s: i for i, s in enumerate(self.symbol_vocab)}
        self.chord_id = {c: i for i, c in enumerate(self.chord_vocab)}
        self.n_sym = len(self.symbol_vocab)
        self.n_chord = len(self.chord_vocab)

    def _build_vocab(self):
        syms, chords = set(), {NEUTRAL_CHORD}
        for i in self.splits["train"]:
            syms.update((V3_DIR / f"{i}_tokens.txt").read_text(encoding="utf-8").split())
            chords.update((V3_DIR / f"{i}_chords.txt").read_text(encoding="utf-8").split())
        symbol_vocab = sorted(syms, key=lambda s: (s.isdigit(), s))   # 数字优先稳定
        if "/" not in symbol_vocab:
            symbol_vocab.append("/")
        return symbol_vocab, sorted(chords)

    # ---------------- split 长序列 ----------------
    def split_sequences(self, name):
        """返回 (token_ids, chord_ids): split 内全部旋律拼接成一条序列。"""
        toks, chds = [], []
        for i in self.splits[name]:
            t = (V3_DIR / f"{i}_tokens.txt").read_text(encoding="utf-8").split()
            c = (V3_DIR / f"{i}_chords.txt").read_text(encoding="utf-8").split()
            toks.extend(t)
            chds.extend(c)
            toks.extend(["/"] * SEP_REPS)
            chds.extend([NEUTRAL_CHORD] * SEP_REPS)
        tok_ids = [self.sym_id[s] for s in toks]
        chord_ids = [self.chord_id.get(c, self.chord_id[NEUTRAL_CHORD]) for c in chds]
        return np.asarray(tok_ids, dtype=np.int32), np.asarray(chord_ids, dtype=np.int32)

    # ---------------- 窗口 ----------------
    def fixed_windows(self, name, n, seed=999):
        """固定随机种子抽 n 个窗口并缓存, 保证所有模型用完全相同的验证/测试窗口。"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"windows_{name}_{n}_seed{seed}.npz"
        if cache.exists():
            z = np.load(cache)
            return z["x"], z.get("c"), z["y"]
        tok_ids, chord_ids = self.split_sequences(name)
        L = len(tok_ids)
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, L - SEQ_LEN, size=n)
        x = np.stack([tok_ids[s:s + SEQ_LEN] for s in starts]).astype(np.int32)
        y = np.array([tok_ids[s + SEQ_LEN] for s in starts], dtype=np.int32)
        c = np.stack([chord_ids[s:s + SEQ_LEN] for s in starts]).astype(np.int32)
        np.savez(cache, x=x, c=c, y=y)
        return x, c, y

    # 训练: 每 epoch 从 train split 随机抽 n 个窗口(覆盖整个训练集)
    def train_windows(self, n, seed):
        tok_ids, chord_ids = self.split_sequences("train")
        L = len(tok_ids)
        starts = np.random.default_rng(seed).integers(0, L - SEQ_LEN, size=n)
        x = np.stack([tok_ids[s:s + SEQ_LEN] for s in starts]).astype(np.int32)
        y = np.array([tok_ids[s + SEQ_LEN] for s in starts], dtype=np.int32)
        c = np.stack([chord_ids[s:s + SEQ_LEN] for s in starts]).astype(np.int32)
        return x, c, y

    def vocab_stats(self):
        print(f"token vocab: {self.n_sym} | chord vocab: {self.n_chord}")
        print("top symbols:", self.symbol_vocab[:8], "...")
        print("chords:", self.chord_vocab)


# ---------------- Markov 基线 ----------------
class MarkovBaseline:
    """k 阶 next-token 语言模型(add-1 + backoff), 不能看到和弦, 作为非神经基线。"""

    def __init__(self, order=6):
        self.order = order
        self.counts = {}

    def fit(self, tok_ids):
        for k in range(1, self.order + 1):
            self.counts[k] = {}
        for i in range(self.order, len(tok_ids)):
            for k in range(1, self.order + 1):
                ctx = tuple(tok_ids[i - k:i])
                self.counts[k][ctx] = self.counts[k].get(ctx, {})
                self.counts[k][ctx][tok_ids[i]] = self.counts[k][ctx].get(tok_ids[i], 0) + 1

    def _prob(self, context, vocab_size, alpha=1.0):
        """backoff: 从最长 context 往下找; add-alpha 平滑。"""
        for k in range(self.order, 0, -1):
            ctx = tuple(context[-k:])
            cnt = self.counts[k].get(ctx)
            if cnt:
                total = sum(cnt.values())
                # 只对见过的 next 做 alpha 平滑, 未见统一小概率(含退回更短)
                probs = {t: (v + alpha) / (total + alpha * (len(cnt) + 1))
                         for t, v in cnt.items()}
                # 保证未知 token 也有一点点概率
                mass = sum(probs.values())
                if mass > 0:
                    return probs, mass
        return {}, 1e-6

    def logprob(self, tok_ids, vocab_size):
        total_nll = 0.0
        n = 0
        for i in range(self.order, len(tok_ids)):
            probs, mass = self._prob(tok_ids[i - self.order:i], vocab_size)
            p = probs.get(tok_ids[i], 1e-8 / vocab_size)
            total_nll += -np.log(max(p * mass, 1e-12))
            n += 1
        return total_nll / max(n, 1)


def tf_windows(x, y, batch=128, c=None, w=None, repeat=False):
    """包装 tf.data; 可选 sample_weight w (逐窗口标量)。"""
    if c is not None:
        if w is not None:
            ds = tf.data.Dataset.from_tensor_slices((x, c, y, w))
            ds = ds.map(lambda a, b, d, ww: ((a, b), d, ww))
        else:
            ds = tf.data.Dataset.from_tensor_slices((x, c, y))
            ds = ds.map(lambda a, b, d: ((a, b), d))
    else:
        if w is not None:
            ds = tf.data.Dataset.from_tensor_slices((x, y, w))
        else:
            ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    return ds
