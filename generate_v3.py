# -*- coding: utf-8 -*-
"""generate_v3: 用实验模型续写流行旋律。

- base:   纯旋律自回归(输入只有旋律 token);
- cond:   给定一条和弦进行(如 C-Am-F-G), 每 0.25 步喂该步所属和弦,
          模型"看着和声"写旋律(可控生成)。
输出: MIDI + 轻量合成 WAV(无外部音源时也能直接听)。
"""
import argparse
import json
import math
from pathlib import Path

import keras
import music21 as m21
import numpy as np

import corpus as C

PROJECT_ROOT = Path(__file__).resolve().parent
EXP_DIR = PROJECT_ROOT / "experiments"
OUT_DIR = PROJECT_ROOT / "out"
TIME_STEP = 0.25

_ROOT12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC_OF = {n: i for i, n in enumerate(_ROOT12)}


def chord_tones(label):
    if label == "N":
        return set()
    if label.endswith("m"):
        r, mode = _PC_OF[label[:-1]], "min"
    else:
        r, mode = _PC_OF[label], "maj"
    return {(r + (3 if mode == "min" else 4)) % 12, (r + 7) % 12, r}


def progression_steps(labels, steps_per_bar=16):
    """把和弦标签重复铺到每个 0.25 步上。"""
    seq = []
    for lab in labels:
        seq.extend([lab] * steps_per_bar)
    return seq


def load_model_and_vocab(name):
    model = keras.models.load_model(str(EXP_DIR / f"{name}.keras"))
    co = C.Corpus()
    return model, co


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def sample(logits_or_probs, temperature=0.6, top_p=0.92, rep_penalty=0.5,
           recent=None, vocab_size=None):
    p = np.asarray(logits_or_probs, dtype=np.float64)
    if abs(p.sum() - 1.0) > 1e-3:            # 传的是 logits
        p = np.log(p + 1e-8)
        p = p / temperature
        p = _softmax(p)
    if recent is not None and rep_penalty > 0:
        logits = np.log(p + 1e-8)
        for tok in set(recent):
            logits[tok] -= rep_penalty
        p = _softmax(logits / temperature)
    order = np.argsort(p)[::-1]
    csum = np.cumsum(p[order])
    keep = int(np.searchsorted(csum, top_p)) + 1
    keep = max(keep, 1)
    mask = np.zeros_like(p, dtype=bool)
    mask[order[:keep]] = True
    p = np.where(mask, p, 0.0)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


def gen_base(model, co, seed_symbols, num_steps=320, temperature=0.6,
             top_p=0.92, rep_penalty=0.6, rep_window=8):
    s2i, i2s = co.sym_id, co.symbol_vocab
    seq = list(seed_symbols)
    hist = [s2i["/"]] * C.SEQ_LEN + [s2i[s] for s in seed_symbols]
    stopped = False
    for _ in range(num_steps):
        win = np.asarray(hist[-C.SEQ_LEN:], dtype=np.int32)[None, :]
        probs = model.predict(win, verbose=0)[0]
        recent = hist[-rep_window:]
        idx = sample(probs, temperature, top_p, rep_penalty, recent)
        sym = i2s[idx]
        if sym == "/":
            stopped = True
            break
        seq.append(sym)
        hist.append(idx)
    return seq, stopped


def gen_cond(model, co, seed_symbols, prog_steps, num_steps=320, temperature=0.6,
             top_p=0.92, rep_penalty=0.6, rep_window=8):
    """prog_steps: 每步一个和弦标签, 可无限取(循环)。"""
    s2i, i2s = co.sym_id, co.symbol_vocab
    c2i, n_chord = co.chord_id, co.n_chord
    seq = list(seed_symbols)
    mhist = [s2i["/"]] * C.SEQ_LEN + [s2i[s] for s in seed_symbols]
    step0 = -(C.SEQ_LEN + len(seed_symbols))        # 全局步计数
    def chord_at(step):
        lab = prog_steps[step % len(prog_steps)] if len(prog_steps) else "N"
        return c2i.get(lab, c2i["N"])
    chist = [chord_at(step0 + j) for j in range(len(mhist))]
    stopped = False
    for _ in range(num_steps):
        mw = np.asarray(mhist[-C.SEQ_LEN:], dtype=np.int32)[None, :]
        cw = np.asarray(chist[-C.SEQ_LEN:], dtype=np.int32)[None, :]
        probs = model.predict([mw, cw], verbose=0)[0]
        idx = sample(probs, temperature, top_p, rep_penalty, mhist[-rep_window:])
        sym = i2s[idx]
        if sym == "/":
            stopped = True
            break
        seq.append(sym)
        mhist.append(idx)
        chist.append(chord_at(step0 + len(mhist) - 1))
    return seq, stopped


# ---------------- 渲染 MIDI / WAV ----------------
def tokens_to_events(tokens):
    events = []
    t = 0.0
    cur = None
    start = 0.0
    for tok in tokens:
        if tok == "_":
            t += TIME_STEP
        elif tok == "/":
            if cur is not None:
                events.append((cur, start, t - start))
            cur = None
            t += TIME_STEP
        else:
            if cur is not None:
                events.append((cur, start, t - start))
            cur = None if tok == "r" else int(tok)
            start = t
            t += TIME_STEP
    if cur is not None and cur != "r":
        events.append((cur, start, t - start))
    return events


def save_midi(events, path):
    s = m21.stream.Stream()
    for pitch, start, dur in events:
        n = m21.note.Note(pitch, quarterLength=dur)
        n.offset = start
        s.insert(n)
    s.write("midi", fp=str(path))


def save_wav(events, path, sr=22050, bpm=112):
    """轻量合成: 正弦+谐波+指数衰减, 便于无音源环境试听。"""
    beat = 60.0 / bpm
    dur_total = (max((e[1] + e[2] for e in events), default=4.0) + 1.0) * beat
    n = int(dur_total * sr)
    buf = np.zeros(n)
    for pitch, start, dur in events:
        freq = 440.0 * 2 ** ((pitch - 69) / 12.0)
        s0 = int(start * beat * sr)
        ln = int(dur * beat * sr)
        if s0 >= n:
            continue
        ln = min(ln, n - s0)
        tt = np.arange(ln) / sr
        env = np.exp(-3.2 * tt / max(dur * beat, 0.05)) * np.minimum(1.0, tt * 60)
        tone = (np.sin(2 * np.pi * freq * tt)
                + 0.5 * np.sin(2 * np.pi * 2 * freq * tt)
                + 0.25 * np.sin(2 * np.pi * 3 * freq * tt))
        buf[s0:s0 + ln] += 0.28 * env * tone
    peak = np.max(np.abs(buf)) or 1.0
    data = (buf / peak * 32000).astype(np.int16)
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="base_s1")
    ap.add_argument("--cond-model", default="cond_s1")
    ap.add_argument("--num-steps", type=int, default=360)
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--seed", type=str, default="74 _ 74 _ 72 _ 71 _ 72 _ 74 _ 76 _ 79")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_symbols = args.seed.split()
    progressions = {
        "Cmaj_Am_Fmaj_G": ["C", "Am", "F", "G"],
        "Am_F_C_G": ["Am", "F", "C", "G"],
        "C_G_Am_F": ["C", "G", "Am", "F"],
    }

    m_base, co = load_model_and_vocab(args.base_model)
    seq, stopped = gen_base(m_base, co, seed_symbols, args.num_steps, args.temperature)
    ev = tokens_to_events(seq)
    save_midi(ev, OUT_DIR / "v3-base-demo.mid")
    save_wav(ev, OUT_DIR / "v3-base-demo.wav")
    print(f"base  : symbols={len(seq)} stopped={stopped} notes={len(ev)}")

    m_cond, co2 = load_model_and_vocab(args.cond_model)
    for pname, plabels in progressions.items():
        psteps = progression_steps(plabels)
        seq, stopped = gen_cond(m_cond, co2, seed_symbols, psteps,
                                args.num_steps, args.temperature)
        ev = tokens_to_events(seq)
        save_midi(ev, OUT_DIR / f"v3-cond-{pname}.mid")
        save_wav(ev, OUT_DIR / f"v3-cond-{pname}.wav")
        print(f"cond {pname}: symbols={len(seq)} stopped={stopped} notes={len(ev)}")
    print("输出目录:", OUT_DIR)


if __name__ == "__main__":
    main()
