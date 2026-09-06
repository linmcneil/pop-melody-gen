# Project Highlights (One-Pager)

## Title
Chord-conditioned LSTM for modern pop lead-melody generation (Pop-K → clean monophonic leads), with a class-rebalanced objective that fixes the "no new notes" degeneracy of vanilla per-token CE.

## One sentence
We turn 305k noisy full-arrangement MIDI loops of Pop-K into 7,000 clean monophonic pop lead melodies (with per-bar inferred chord labels), train a dual-stream LSTM that can keep writing a melody under a user-chosen chord progression, and evaluate it with an honest by-song train/val/test split against n-gram Markov, single-stream LSTM, and Transformer baselines.

## What we did
1. Data engineering: melody-layer extraction from polyphonic loops, 16th-grid quantization, key/octave normalization, per-bar chord inference from the accompaniment, 8:1:1 by-song split.
2. Found a real failure mode: vanilla sparse-CE models maximize likelihood by never emitting note onsets (note top-1 acc ≈ 2–3% ≈ random; sampling yields holds/end only).
3. Fixed it at training time with note-onset reweighting (λ=6): note-onset accuracy jumps to ≈28–30%, sampling produces notes again.
4. Models (CPU-trainable, ~0.2–0.3M params): Markov-6, single-stream LSTM, dual-stream chord-conditioned LSTM, 2-layer Transformer; all share fixed validation/test windows.

## Key numbers (final, seed 1, Windows/CPU)
Data: 7,000 melodies | 49-symbol melody vocab | 25 chord labels | split 5600/700/700 (by song).

Test-set next-token metrics on the same 4,000 fixed windows (unweighted CE for all):

| model | test NLL | acc | PPL | note-onset acc |
|---|---|---|---|---|
| Markov-6 | 0.916 | 0.766 | 2.50 | — |
| LSTM (vanilla CE) | 0.809 | 0.752 | 2.25 | 0.03 |
| LSTM+chords (vanilla CE) | 0.824 | 0.751 | 2.28 | 0.02 |
| **LSTM + note reweight** | 0.940 | 0.660 | 2.56 | **0.28** |
| **LSTM+chords + reweight** | 0.943 | 0.661 | 2.57 | **0.30** |
| Transformer-2L (vanilla CE) | **0.783** | 0.755 | **2.19** | 0.07 |

Interpretation: standard NLL rewards the "always hold/end" lazy solution; the rebalanced objective trades ~0.13 NLL for the ability to actually compose notes. On raw likelihood the Transformer is best; on "can the model keep writing under a progression" the rebalanced chord model wins.

Generation (12 fixed test-split prompts, temp 0.7, C–Am–F–G):

| model | avg len (tokens) | total notes | natural stop | rep(8-gram) | pitch entropy | range |
|---|---|---|---|---|---|---|
| LSTM reweight | 25 | 135 | 1.00 | 0.014 | 1.60 | 6.2 |
| **LSTM+chords reweight** | **76** | **426** | 0.75 | 0.094 | 2.02 | 9.9 |
| LSTM vanilla | 12 | 25 | 1.00 | 0.125 | 1.18 | 2.5 |
| LSTM+chords vanilla | 4 | 6 | 1.00 | — | 0.40 | 0.4 |

Chords here are an accompaniment given to the model: conditioning on them roughly triples the continuation length and note output, widens the range, and raises melodic interest — while the user controls the progression (demos for C–Am–F–G, Am–F–C–G, C–G–Am–F included).

## Human listening
Protocol + stimulus generator (A = original human melody, B = LSTM, C = chord LSTM, same loop/progression) are in `HUMAN_EVAL.md` + `human_eval/`; 4 example trios already rendered under `human_eval/stimuli/`.

## Reproduce
See README.md: `preprocess_v3.py` → `train_exp.py --kind base|cond|xformer --note-weight N` → `analyze.py` + `eval_v3.py`.

## Ethics & license
Pop-K is CC BY-NC (non-commercial research); code is MIT. Cite the dataset + POP909 paper (see CITATIONS.bib).