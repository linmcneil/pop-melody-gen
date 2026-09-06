# -*- coding: utf-8 -*-
"""Preprocess Pop-K MIDI loops into melody token sequences.

为什么不是直接吃 MIDI:
  Pop-K 每条 .mid 是一段 8 小节现代流行 loop, 但它的"一条音轨"里其实把
  低音 + 和弦 + 高音主旋律(lead)全部叠在了一起。所以预处理的核心步骤是
  "主旋律层抽取(lead extraction)": 逐时刻比较正在发声的所有音符,
  只有当新音符高于当前已响的全部音符、且与同刻齐奏的音符拉开 >=6 个半音的
  音区差时, 才把它判为主旋律; 其余低音/和弦被丢弃。主旋律音符之间如重叠
  则截断成单声部。这个启发式对"顶部领奏 + 块状伴奏"的现代流行 loop 很有效。

之后沿用 v1/v2 的符号化:
  1) 音符/休止都落在 0.25 拍网格上(16 分音符);
  2) 按 Krumhansl 相关把每条旋律统一移到 C 大调 / A 小调;
  3) 编码成 [pitch][r][_][/] 序列(pitch=MIDI 号, _=延音, r=休止);
  4) 输出与 v2 兼容: dataset/ 单曲文本 + file_dataset + mapping.json。

运行: python preprocess.py
"""
import gzip
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import mido
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

TAR_PATH = PROJECT_ROOT / "data" / "popk_300k_mid.tar"
TAR_GZ_PATH = PROJECT_ROOT / "data" / "popk_300k_mid.tar.gz"
SAVE_DIR = PROJECT_ROOT / "dataset"
SINGLE_FILE_DATASET = PROJECT_ROOT / "file_dataset"
MAPPING_PATH = PROJECT_ROOT / "mapping.json"

SEQUENCE_LENGTH = 64          # 与 v2 一致: 一条训练序列的步数(4 小节 @0.25拍)
TIME_STEP = 0.25              # 一个时间步 = 16 分音符

MAX_CANDIDATES = 90000        # 最多扫描多少个 popk 文件
TARGET_SONGS = 7000           # 目标保留的主旋律条数
TAR_START = 0                 # 从 tar 第几个成员开始顺序扫描
MIN_NOTES = 8                 # 主旋律最少音符数
MAX_NOTES = 72                # 主旋律最多音符数
PITCH_MIN = 48                # 移调后允许的最低音
PITCH_MAX = 103               # 移调后允许的最高音

# Krumhansl-Kessler 调性模板(C 大调 / A 小调, pitch-class 0..11)
K_K_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
K_K_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


# --------------------------------------------------------------------------
# 1. MIDI 读取与主旋律层抽取
# --------------------------------------------------------------------------
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


def extract_lead(ppq, notes):
    """从"低音+和弦+主旋律"叠好的轨里抽单声部主旋律。

    返回 lead=[onset_tick, pitch, end_tick], 按 onset 升序, 已被截断成单声部。
    """
    events = sorted(notes, key=lambda x: (x[0], x[2], x[1]))
    active = []                # [end_tick, pitch] 当前仍在响的所有音(含主旋律)
    lead = []                  # [onset_tick, pitch, end_tick]
    i, n = 0, len(events)
    while i < n:
        t = events[i][0]
        active = [(e, p) for e, p in active if e > t]
        group = []
        while i < n and events[i][0] == t:
            group.append(events[i])
            i += 1
        prev_max = max((p for _, p in active), default=-1000)
        group.sort(key=lambda g: g[1], reverse=True)
        top = group[0]
        second = group[1][1] if len(group) > 1 else -1000
        # 规则: 新音必须高于所有"已在响"的音; 若是齐奏和弦, 还要和次高音拉开 >=6 半音,
        #       否则一律当伴奏。这能同时滤掉"和弦顶音冒充主旋律"。
        if top[1] > prev_max and (len(group) == 1 or (top[1] - second) >= 6):
            if lead and lead[-1][2] > t:      # 前一个主旋律若还响着, 截断成单声部
                lead[-1][2] = t
            lead.append([t, top[1], t + top[2]])
            active.append([t + top[2], top[1]])
        active.extend([[g[0] + g[2], g[1]] for g in group])
    return lead


# --------------------------------------------------------------------------
# 2. 量化 + 移调
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# 3. 符号编码(与 v1/v2 相同的时间步展开)
# --------------------------------------------------------------------------
def encode_lead(lead):
    """把量化+移调后的 [start_step,pitch,end_step] 展开成符号序列。"""
    if not lead:
        return []
    lead = sorted(lead)
    tokens = []
    prev_end = lead[0][0]
    for start, pitch, end in lead:
        if start > prev_end:                          # 休止
            tokens.append("r")
            tokens.extend(["_"] * (start - prev_end - 1))
        tokens.append(str(pitch))
        tokens.extend(["_"] * (end - start - 1))
        prev_end = max(prev_end, end)
    return tokens


# --------------------------------------------------------------------------
# 4. 主流程
# --------------------------------------------------------------------------
def process_one(midi_bytes):
    """单文件: 抽取->量化->移调->编码。失败/不过滤返回 None。"""
    try:
        ppq, notes = read_notes(midi_bytes)
    except Exception:
        return None
    if not notes:
        return None
    lead = extract_lead(ppq, notes)
    if not (MIN_NOTES <= len(lead) <= MAX_NOTES):
        return None
    qlead = quantize_lead_to_grid(lead, ppq)
    if not (MIN_NOTES <= len(qlead) <= MAX_NOTES):
        return None
    # 总长度: 应接近 8 小节(128 步 @0.25), 允许前奏/尾音稍有出入
    total_steps = qlead[-1][1] - qlead[0][0] + 1
    if total_steps > 40 * 4:
        return None
    shift = estimate_key_shift(qlead)
    qlead = transpose(qlead, shift)
    qlead = normalize_octave(qlead, target_center=74)
    pitches = [p for _, p, _ in qlead]
    if max(pitches) > PITCH_MAX or min(pitches) < PITCH_MIN:
        return None
    return encode_lead(qlead)


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


def preprocess(tar_path=None):
    if tar_path is None:
        tar_path = ensure_tar()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    tf = tarfile.open(str(tar_path))
    members = tf.getmembers()
    total_members = len(members)
    if TAR_START >= total_members:
        raise ValueError("TAR_START 超出成员总数")

    saved, tried, skipped = 0, 0, 0
    print(f"共 {total_members} 个 MIDI, 从 #{TAR_START} 顺序扫描, 目标 {TARGET_SONGS} 条主旋律")
    idx = TAR_START
    while saved < TARGET_SONGS and tried < MAX_CANDIDATES and idx < total_members:
        m = members[idx]
        idx += 1
        tried += 1
        midi_bytes = tf.extractfile(m).read()
        tokens = process_one(midi_bytes)
        if tokens is None or len(tokens) < MIN_NOTES:
            skipped += 1
            continue
        (SAVE_DIR / str(saved)).write_text(" ".join(tokens), encoding="utf-8")
        saved += 1
        if saved % 500 == 0:
            print(f"  已保存 {saved} 条, 扫描 {tried}, 跳过 {skipped}")
    tf.close()

    if saved == 0:
        raise RuntimeError("一条都没留下, 检查抽取/过滤规则")

    print(f"预处理完成: 保存 {saved} 条可用主旋律 (扫描 {tried} 个文件, 跳过 {skipped})")
    return saved


def create_single_file_dataset(save_dir=SAVE_DIR):
    if not save_dir.exists():
        raise FileNotFoundError(f"找不到 {save_dir}, 请先运行 preprocess()")
    songs_text = ""
    for path in sorted(save_dir.iterdir()):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            songs_text += content + " " + "/ " * SEQUENCE_LENGTH
    songs_text = songs_text.rstrip()
    SINGLE_FILE_DATASET.write_text(songs_text, encoding="utf-8")
    tokens = songs_text.split()
    print(f"拼接完成: 总符号数 {len(tokens)} -> {SINGLE_FILE_DATASET.name}")
    return songs_text


def create_mapping(songs_text):
    vocabulary = sorted(set(songs_text.split()))
    mapping = {symbol: index for index, symbol in enumerate(vocabulary)}
    MAPPING_PATH.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"词表大小: {len(mapping)} -> {MAPPING_PATH}")
    return mapping


def load_mapping():
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def build_windows(sequence_length=SEQUENCE_LENGTH, max_samples=None):
    """与 v2 完全一致的滑窗切分。"""
    text = SINGLE_FILE_DATASET.read_text(encoding="utf-8")
    if not text.strip():
        raise FileNotFoundError(f"{SINGLE_FILE_DATASET} 为空, 请先预处理")
    tokens = text.split()
    mapping = load_mapping()
    int_tokens = [mapping[symbol] for symbol in tokens]
    num_sequences = len(int_tokens) - sequence_length
    if max_samples is not None:
        num_sequences = min(num_sequences, int(max_samples))
    x_int, y_int = [], []
    for i in range(num_sequences):
        x_int.append(int_tokens[i:i + sequence_length])
        y_int.append(int_tokens[i + sequence_length])
    print(f"共生成 {num_sequences} 条训练序列, 每条 {sequence_length} 步")
    return np.asarray(x_int, dtype=np.int32), np.asarray(y_int, dtype=np.int32)


def main():
    n = preprocess()
    songs_text = create_single_file_dataset()
    create_mapping(songs_text)
    print(f"完成: {n} 条流行主旋律已就绪")


if __name__ == "__main__":
    main()
