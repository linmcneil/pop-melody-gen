# Human Evaluation Protocol (Melody Quality)

Goal: give a *protocol* number that is reproducible and honest, not "we think it sounds good".
We do **not** claim we already ran a full-scale listening test; this document is the exact
protocol to run it (target N >= 20 raters, e.g., classmates).

## Stimuli
- Source: 12 randomly chosen 8-bar loops from the Pop-K test split (never seen in training).
- Condition A "pop human": the original lead melody rendered from the same preprocessed corpus.
- Condition B "base": LSTM without chord conditioning.
- Condition C "chord": chord-conditioned LSTM under the loop's own inferred progression.
- All rendered as MIDI->WAV through the same lightweight synth (no timbre cue differences),
  same tempo (112 BPM), similar loudness, ~8-16 seconds each.

## Task
Each rater listens to 3 versions of the same chord progression (A/B/C) and rates on 1-5:
1. Melodiousness (旋律感): does it sound like a singable melody?
2. Harmonic fit (和声贴合): do the notes fit the chords / not clash?
3. Naturalness / musicality (是否像人写的流行主旋律)?
4. Novelty (是否有新意, 不照抄原曲)?
Then a forced-choice: "which is the most plausible pop vocal lead?"

## Analysis
- Report mean + std per condition and per question.
- Friedman test (non-parametric, repeated measures) across A/B/C, then Wilcoxon signed-rank
  post-hoc with Bonferroni correction for B vs C.
- Report effect size (e.g., matched rank-biserial or Cohen d_z) and inter-rater agreement
  (Krippendorff alpha) to justify aggregation.

## Files (implemented)
- Stimulus generator: `human_eval/make_stimuli.py` (renders A/B/C trios from the test split
  under each loop's own inferred progression; 4 trios already rendered in
  `human_eval/stimuli/{A,B,C}/`).
- Analysis: `human_eval/analysis.py` (means/sd per condition, Friedman + Wilcoxon with
  Bonferroni, matched rank-biserial; pure numpy).
- After running: human_eval/stimuli/{A,B,C}/..., human_eval/results.csv
  (rater,loop,A_mel,B_mel,C_mel,A_harm,B_harm,C_harm,A_nat,B_nat,C_nat,A_nov,B_nov,C_nov,choice),
  then `python human_eval/analysis.py --csv human_eval/results.csv`.
