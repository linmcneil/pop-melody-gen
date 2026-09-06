# Sample Statement-of-Purpose Paragraph (adjust to your program)

> Symbolic music generation sits at the intersection of sequence modeling and human
> creativity, yet most published datasets in the area are classical or folk. In my project
> "Chord-conditioned LSTM for Modern Pop Lead-Melody Generation", I took on the messy side
> of the problem: real pop MIDI collections are noisy — the file officially called a
> "melody track" usually packs bass, chords and lead into one polyphonic track, and when
> melody is quantized to a 16th-note grid, plain cross-entropy training quietly degenerates
> into a model that always predicts holds/end and almost never attacks new notes
> (measured note-onset top-1 accuracy ≈3%, near random). I built a full pipeline that
> extracts clean monophonic leads (7,000 of them from 305k Pop-K loops), infers a per-bar
> chord label from the remaining accompaniment, splits the corpus by song (5600/700/700),
> and trains five next-token models — Markov-6, a single-stream LSTM, a dual-stream
> chord-conditioned LSTM, a note-reweighted variant of each, and a small causal
> Transformer — all sharing the same fixed validation/test windows. On the standard metric
> the Transformer is strongest (test NLL 0.783, PPL 2.19 vs 2.50 for Markov), but the more
> instructive finding is the rebalancing fix: note-onset accuracy rises from ~3% to 28–30%,
> and the chord-conditioned model then keeps writing much longer, denser melodies under a
> user-chosen progression (C–Am–F–G): 76 tokens and 426 notes across 12 prompts vs 25 tokens
> and 135 notes without chords. I also designed a human A/B/C listening protocol and
> rendered the stimulus set. This project taught me to run a small end-to-end study:
> data forensics before modeling, splits that prevent leakage, baselines that make
> "improvement" measurable, and honest reporting of what remains unsolved — full-song
> verse/chorus structure and scale-up to multi-instrument tracks.

# Suggested CV bullets (EN)

- Built an end-to-end symbolic-music pipeline: melody-layer extraction + key/octave
  normalization + per-bar chord inference over 305k pop MIDI loops → 7,000 clean
  monophonic lead melodies (8:1:1 by-song split).
- Diagnosed a class-imbalance failure of per-token CE on grid-encoded melody (note-onset
  top-1 acc ≈3%, ≈ random) and fixed it with note-onset reweighting (→ 28–30%),
  benchmarking Markov-6 / LSTM / chord-conditioned LSTM / Transformer on identical fixed
  validation/test windows (test NLL 0.783–0.943, PPL 2.19–2.57).
- Showed chord conditioning lets a small CPU-trained LSTM write longer, denser, wider-range
  pop melodies under a user-chosen progression (C–Am–F–G demos: 426 vs 135 notes over 12
  prompts) and shipped a human A/B/C listening protocol with rendered stimuli.

# 中文要点(写在简历/作品集里)

- 技术栈:Python · TensorFlow/Keras · music21/mido · MIDI/信号处理 · 统计评测(实验设计)。
- 亮点:不是“跑通一个 demo”,而是发现并修复了“网格化旋律 + vanilla CE 不会出音”的方法论问题,
  并用同一固定窗口协议公平对比 5 个模型;附可试听 MIDI/WAV 与听感评测材料。
- 一句话:数据清洗(从整轨伴奏提取人声主旋律 + 逐小节和弦)、序列建模、统计评测全链路自己完成。