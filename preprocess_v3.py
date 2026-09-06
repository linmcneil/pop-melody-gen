# -*- coding: utf-8 -*-
"""preprocess_v3: 在旋律抽取基础上输出 逐token 对齐的和弦条件数据 + 按歌划分。

产出(写入 dataset_v3/):
  <i>_tokens.txt   旋律 token 序列(与 v2 相同编码)
  <i>_chords.txt   与每个 token 对齐的和弦标签(按 0.25 步的绝对小节号映射)
  splits.json      按歌(melody)id 的 train/val/test 划分(不泄漏窗口)
  stats.json       语料统计
"""
import gzip
import io
import json
import os
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import mido
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# Low-level helpers shared with the v1/v2 pipeline (inlined here so
# preprocess_v3.py is self-contained): MIDI read / grid quantize /
# Krumhansl key estimation / octave normalization / tar bootstrap.
# ------------------------------------------------------------------
TIME_STEP = 0.25              # 一个时间步 = 16 分音符
TAR_PATH = PROJECT_ROOT / "data" / "popk_300k_mid.tar"
TAR_GZ_PATH = PROJECT_ROOT / "data" / "popk_300k_mid.tar.gz"
K_K_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
K_K_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def read_notes(midi_bytes):
    """解析 MIDI, 返回 (ppq, notes); notes=[onset_ticks, pitch, dur_ticks]。"""
    mf = mido.MidiFile(file=io.BytesIO(midi_bytes))
    ppq = mf.ticks_per_beat
    notes, active = [], {}
    for track in mf.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.channel < 9:
                active[(msg.channel, msg.note)] = t
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    notes.append([active[key], msg.note, t - active[key]])
                    del active[key]
    return ppq, notes


def quantize_lead_to_grid(lead, ppq):
    """把主旋律(ticks)对齐到 0.25 拍网格。

    用"把 start/end 各round到 grid 的整数步"实现, 返回按 onset 升序的
    [start_step, pitch, end_step]; 单声部、end>=start。
    """
    steps_per_beat = int(round(1.0 / TIME_STEP))          # 4
    q = []
    for onset, pitch, end in sorted(lead):
        s0 = int(round(onset * steps_per_beat / ppq))
        s1 = int(round(end * steps_per_beat / ppq))
        if s1 <= s0:                                       # 量化后过短, 丢弃
            continue
        q.append([s0, pitch, s1])
    # 单声部保证: 后一个音不能在前一个音结束前开始(容差已经由网格吸收)
    out = []
    for note in q:
        if out and note[0] < out[-1][2]:
            note[0] = out[-1][2]
        if note[1] <= note[2]:
            out.append(note)
    return out


def _corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1])


def estimate_key_shift(lead):
    """对主旋律做 Krumhansl 相关, 返回移调到 C 大调 / A 小调所需的半音数。"""
    hist = np.zeros(12)
    for start, pitch, end in lead:
        hist[int(pitch) % 12] += max(end - start, 1)
    hist = hist / hist.sum()
    best_key, best_score = None, -1e9
    for tonic in range(12):
        for mode, profile in (("major", K_K_MAJOR), ("minor", K_K_MINOR)):
            score = _corr(hist, _rotate(profile, tonic))
            if score > best_score:
                best_score = score
                best_key = (tonic, mode)
    tonic, mode = best_key
    target_pc = 0 if mode == "major" else 9       # C 大调 / A 小调
    shift = (target_pc - tonic) % 12
    if shift > 6:
        shift -= 12
    return shift


def _rotate(vec, k):
    k = k % len(vec)
    return vec[k:] + vec[:k]


def transpose(lead, shift):
    return [[s0, int(p) + shift, s1] for s0, p, s1 in lead]


def normalize_octave(lead, target_center=74):
    """八度归一: 把整条旋律按中位数移动到统一音区, 跨八度变体共享同一套统计。"""
    pitches = [p for _, p, _ in lead]
    median = float(np.median(pitches))
    best = None
    for octave in range(-2, 3):
        shift = 12 * octave
        score = abs((median + shift) - target_center)
        if best is None or score < best[0]:
            best = (score, shift)
    return transpose(lead, best[1])


def ensure_tar():
    """如果只有 .tar.gz, 先流式解压成普通 tar(gzip tar 随机读很慢)。"""
    if TAR_PATH.exists():
        return TAR_PATH
    if TAR_GZ_PATH.exists():
        print("解压 popk_300k_mid.tar.gz -> popk_300k_mid.tar ...")
        with gzip.open(str(TAR_GZ_PATH), "rb") as src, open(str(TAR_PATH), "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        print("解压完成")
        return TAR_PATH
    raise FileNotFoundError("找不到 data/popk_300k_mid.tar(.gz), 请先下载语料")


OUT_DIR = PROJECT_ROOT / "dataset_v3"

N_BARS = 8
STEPS_PER_BAR = 16          # 4 拍 * (0.25 拍/步)
NOTES_PER_BAR = 4
MAX_CANDIDATES = 90000
TARGET_SONGS = 7000
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 2025
MIN_NOTES = 8
MAX_NOTES = 72
PITCH_MIN = 48
PITCH_MAX = 103

_ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]




def split_parts(ppq, notes):
    """返回 (lead, accomp): 与 extract_lead 相同的判定, 但保留伴奏音符。"""
    events = sorted(notes, key=lambda x: (x[0], x[2], x[1]))
    active = []   # [end_tick, pitch]
    lead = []
    accomp = []
    i, n = 0, len(events)
    while i < n:
        t = events[i][0]
        active = [(e, p) for e, p in active if e > t]
        group = []
        while i < n and events[i][0] == t:
            group.append(events[i]); i += 1
        prev_max = max((p for _, p in active), default=-1000)
        group.sort(key=lambda g: g[1], reverse=True)
        top = group[0]
        second = group[1][1] if len(group) > 1 else -1000
        is_lead = top[1] > prev_max and (len(group) == 1 or (top[1] - second) >= 6)
        if is_lead:
            if lead and lead[-1][2] > t:
                lead[-1][2] = t
            lead.append([t, top[1], t + top[2]])
            active.append([t + top[2], top[1]])
        else:
            accomp.append([t, top[1], t + top[2]])
        active.extend([[g[0] + g[2], g[1]] for g in group])
    return lead, accomp


def infer_chord_labels(accomp, ppq, shift, n_bars=N_BARS):
    """伴奏层 -> 每小节一个 (root, mode) 标签; 根音取小节内最低音(贝斯)。"""
    pcs_per_bar = [Counter() for _ in range(n_bars)]
    low_per_bar = [None] * n_bars
    for onset, pitch, end in accomp:
        p = pitch + shift
        s = onset / ppq
        e = end / ppq
        for b in range(n_bars):
            bs, be = b * NOTES_PER_BAR, (b + 1) * NOTES_PER_BAR
            ov = min(e, be) - max(s, bs)
            if ov > 0.2:                     # 与这个节重合超过 0.2 拍才算
                pcs_per_bar[b][p % 12] += ov
                if low_per_bar[b] is None or p < low_per_bar[b]:
                    low_per_bar[b] = p

    labels = []
    prev = None
    for b in range(n_bars):
        pcs = pcs_per_bar[b]
        low = low_per_bar[b]
        if not pcs or low is None:
            labels.append(prev if prev else "N")
            prev = labels[-1]
            continue
        root = low % 12
        w3 = pcs[(root + 3) % 12]
        w4 = pcs[(root + 4) % 12]
        if w4 > w3 * 1.2:
            mode = "maj"
        elif w3 > w4 * 1.2:
            mode = "min"
        else:
            mode = "min" if (prev or "").endswith("m") and w3 > 0 else "maj"
        label = _ROOT_NAMES[root] if mode == "maj" else _ROOT_NAMES[root] + "m"
        labels.append(label)
        prev = label
    return labels


def encode_with_chords(qlead, chord_labels):
    """qlead: [start_step,pitch,end_step] 移调/八度归一后; 返回 (tokens, chords_per_token)。

    每个 token 恰好占一个 0.25 步: 第一个 token 在 step=f, 之后顺延。于是
    token 的绝对 step 已知, 用 chord_labels[step // 16] 取该步所在小节的和弦。
    """
    if not qlead:
        return [], []
    qlead = sorted(qlead)
    f = qlead[0][0]

    def chord_for(step):
        bar = min(step // STEPS_PER_BAR, len(chord_labels) - 1)
        return chord_labels[bar]

    tokens, chords = [], []
    cur = f
    prev_end = f
    for start, pitch, end in qlead:
        while cur < start:                       # 前一个音结束到本音开始: 休止
            tokens.append("r" if cur == prev_end else "_")
            chords.append(chord_for(cur))
            cur += 1
        tokens.append(str(pitch))                # 本音
        chords.append(chord_for(cur))
        cur += 1
        while cur < end:                         # 延音
            tokens.append("_")
            chords.append(chord_for(cur))
            cur += 1
        prev_end = max(prev_end, end)
    return tokens, chords


def process_one(midi_bytes):
    """同 preprocess.process_one, 但额外返回 (tokens, per-token chord labels)。"""
    try:
        ppq, notes = read_notes(midi_bytes)
    except Exception:
        return None
    if not notes:
        return None
    lead, accomp = split_parts(ppq, notes)
    if not (MIN_NOTES <= len(lead) <= MAX_NOTES):
        return None
    qlead = quantize_lead_to_grid(lead, ppq)
    if not (MIN_NOTES <= len(qlead) <= MAX_NOTES):
        return None
    total_steps = qlead[-1][1] - qlead[0][0] + 1
    if total_steps > 40 * 4:
        return None
    shift = estimate_key_shift(qlead)
    qlead = transpose(qlead, shift)
    qlead = normalize_octave(qlead, target_center=74)
    pitches = [p for _, p, _ in qlead]
    if max(pitches) > PITCH_MAX or min(pitches) < PITCH_MIN:
        return None
    if not accomp:
        return None
    chord_labels = infer_chord_labels(accomp, ppq, shift, N_BARS)
    tokens, chords = encode_with_chords(qlead, chord_labels)
    if len(tokens) < MIN_NOTES or len(chords) != len(tokens):
        return None
    return tokens, chords


def run_preprocess():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = ensure_tar()
    tf = tarfile.open(str(tar_path))
    members = tf.getmembers()
    total_members = len(members)

    saved, tried, skipped = 0, 0, 0
    idx = 0
    print(f"扫描 {tar_path.name} 生成旋律+和弦数据集(目标 {TARGET_SONGS} 条)...")
    n_bad_chords = 0
    while saved < TARGET_SONGS and tried < MAX_CANDIDATES and idx < total_members:
        m = members[idx]; idx += 1; tried += 1
        res = process_one(tf.extractfile(m).read())
        if res is None:
            skipped += 1
            continue
        tokens, chords = res
        if len(set(chords)) < 2:
            n_bad_chords += 1
            skipped += 1
            continue
        (OUT_DIR / f"{saved}_tokens.txt").write_text(" ".join(tokens), encoding="utf-8")
        (OUT_DIR / f"{saved}_chords.txt").write_text(" ".join(chords), encoding="utf-8")
        saved += 1
        if saved % 500 == 0:
            print(f"  已保存 {saved} 条 (扫描 {tried}, 跳过 {skipped})")
    tf.close()
    if saved == 0:
        raise RuntimeError("没有生成任何旋律")
    print(f"完成: 保存 {saved} 条旋律+和弦 (扫描 {tried}, 跳过 {skipped})")
    return saved


def assign_splits(n_songs):
    rng = np.random.default_rng(SPLIT_SEED)
    ids = list(range(n_songs))
    rng.shuffle(ids)
    n_val = int(n_songs * SPLIT["val"])
    n_test = int(n_songs * SPLIT["test"])
    splits = {
        "train": sorted(ids[: n_songs - n_val - n_test]),
        "val": sorted(ids[n_songs - n_val - n_test: n_songs - n_test]),
        "test": sorted(ids[n_songs - n_test:]),
    }
    meta = {"seed": SPLIT_SEED, "n_train": len(splits["train"]),
            "n_val": len(splits["val"]), "n_test": len(splits["test"])}
    (OUT_DIR / "splits.json").write_text(json.dumps(meta), encoding="utf-8")
    for name, arr in splits.items():
        (OUT_DIR / f"split_{name}.txt").write_text("\n".join(map(str, arr)), encoding="utf-8")
    print(f"划分: {meta}")
    return splits


def collect_stats():
    lens = []
    chord_cnt = Counter()
    for i in range(TARGET_SONGS):
        toks = (OUT_DIR / f"{i}_tokens.txt").read_text(encoding="utf-8").split()
        chs = (OUT_DIR / f"{i}_chords.txt").read_text(encoding="utf-8").split()
        lens.append(len(toks))
        chord_cnt.update(chs)
    stats = {
        "n_songs": TARGET_SONGS,
        "token_len_p10_p50_p90": [int(np.percentile(lens, q)) for q in (10, 50, 90)],
        "mean_tokens": float(np.mean(lens)),
        "top_chords": chord_cnt.most_common(20),
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stats)


def main():
    n = run_preprocess()
    assign_splits(n)
    collect_stats()


if __name__ == "__main__":
    main()
