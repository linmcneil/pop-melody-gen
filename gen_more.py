"""批量生成流行旋律并保存 MIDI (v2 sampler 为主, 附带 1 首 naive 对照)"""
import numpy as np
import generate as G

seeds = [
    "72 _ 72 _ 74 _ 76 _ 74 _ 72 _ _ _ _",
    "79 _ 79 _ 76 _ 74 _ 72 _ _ 71 _ 69 _ 71 _",
    "76 _ 74 _ 71 _ 74 _ 76 _ 79 _ 81 _ 79 _ 76",
    "84 _ 81 _ 79 _ 76 _ _ 74 _ 72 _ _ 71 _ 72 _",
    "72 _ 76 _ 79 _ 76 _ 74 _ 76 _ 72 _ _ _ _",
    "83 _ 81 _ 79 _ 76 _ 74 _ 71 _ 74 _ 76 _ _",
]

model, mapping, symbols, vs = G.load_model_and_mapping(r"C:\Users\lzq13\pop-melody-gen\model.keras")
print("vocab size", vs)

for idx, seed in enumerate(seeds, 1):
    seed_symbols = seed.split()
    for tok in seed_symbols:
        if tok not in mapping:
            raise SystemExit(f"seed token not in vocab: {tok}")
    melody, stopped = G.generate_melody(
        model, mapping, symbols, seed_symbols, kind="int", sampler="v2",
        num_steps=360, temperature=0.5, top_p=0.92,
        repetition_penalty=0.5, rep_window=8,
    )
    out = rf"C:\Users\lzq13\pop-melody-gen\pop-mel-{idx}.mid"
    G.save_melody(melody, file_path=out)
    print(f"[{idx}] symbols={len(melody)} stopped_by_slash={stopped}")
    print("   tail:", " ".join(melody[-40:]))
    print("   onset:", [t for t in melody if t not in ("_","r","/")][:24])

# naive 对照
seed_symbols = seeds[0].split()
melody, stopped = G.generate_melody(
    model, mapping, symbols, seed_symbols, kind="int", sampler="naive",
    num_steps=360, temperature=0.5,
)
G.save_melody(melody, file_path=r"C:\Users\lzq13\pop-melody-gen\pop-mel-naive.mid")
print("naive saved", len(melody), stopped)
