# -*- coding: utf-8 -*-
"""生成 HUMAN_EVAL.md 所述 A/B/C 三条件听感材料。

A = 真实人声主旋律(test split 原曲, 未进训练);
B = base LSTM(纯旋律自回归)从同一 prompt 续写;
C = chord-conditioned LSTM 从同一 prompt、按原曲自己推得的和弦进行续写。

同一轻量合成器渲染, 无音色差异; 固定速度 112 BPM。材料放在
human_eval/stimuli/{A,B,C}/ 下, 供 N>=20 名被试按 HUMAN_EVAL.md 打分。

用法:
  python human_eval/make_stimuli.py --n-loops 6 --out human_eval/stimuli
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import keras

import corpus as C
import generate_v3 as G

EXP_DIR = PROJECT_ROOT / "experiments"
N_BARS = 8
PROMPT_LEN = 24
TEMP = 0.55


def loop_chords(co, i, n_bars=N_BARS):
    """取该 loop 前 n_bars 小节、每小节第一拍的和弦作为进行(原曲自带)。"""
    chds = (C.V3_DIR / f"{i}_chords.txt").read_text(encoding="utf-8").split()
    return [chds[b * 16] for b in range(min(n_bars, len(chds) // 16))]


def render(kind, tokens, out_path):
    ev = G.tokens_to_events(tokens)
    G.save_midi(ev, out_path)
    G.save_wav(ev, out_path.with_suffix(".wav"))
    return len(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loops", type=int, default=6)
    ap.add_argument("--num-steps", type=int, default=N_BARS * 16)
    ap.add_argument("--base", default="base_s1")
    ap.add_argument("--cond", default="cond_s1")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "human_eval" / "stimuli"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    co = C.Corpus()
    # 取 test split 前 N 条足够长、含和弦的主旋律(可换成随机抽样)
    pool = [i for i in co.splits["test"]
            if len((C.V3_DIR / f"{i}_tokens.txt").read_text(encoding="utf-8").split()) >= 64]
    loops = pool[:args.n_loops]
    assert len(loops) == args.n_loops, f"只有 {len(pool)} 条够长"

    m_base = keras.models.load_model(str(EXP_DIR / f"{args.base}.keras"))
    m_cond = keras.models.load_model(str(EXP_DIR / f"{args.cond}.keras"))
    out = Path(args.out)
    for tag in ("A", "B", "C"):
        (out / tag).mkdir(parents=True, exist_ok=True)

    for k, i in enumerate(loops):
        toks = (C.V3_DIR / f"{i}_tokens.txt").read_text(encoding="utf-8").split()
        seed = toks[:PROMPT_LEN]
        chords = loop_chords(co, i)
        stem = f"loop{k + 1:02d}"

        # A: 真实主旋律(整条)
        nA = render("A", toks, out / "A" / f"{stem}-A.mid")
        # B: base 续写
        seqB, stopB = G.gen_base(m_base, co, seed, args.num_steps, TEMP)
        nB = render("B", seqB, out / "B" / f"{stem}-B.mid")
        # C: cond 按原曲和弦续写
        psteps = G.progression_steps(chords)
        seqC, stopC = G.gen_cond(m_cond, co, seed, psteps, args.num_steps, TEMP)
        nC = render("C", seqC, out / "C" / f"{stem}-C.mid")
        print(f"{stem} (id={i}): A notes={nA} | B notes={nB} stop={stopB} | "
              f"C notes={nC} stop={stopC} chords={'/'.join(chords)}")

    print("stimuli ->", out)


if __name__ == "__main__":
    main()