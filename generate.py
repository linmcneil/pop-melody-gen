"""用模型续写旋律:支持两种采样策略,并导出 MIDI。

naive 采样(对照 v1 的原始做法):
  直接把 softmax 概率做温度缩放后按多项式分布采样;
  问题:容易陷入“单音无限重复/复读机”,也不控制是否结束。

v2 采样(本项目的小创新):
  1. temperature 温度控制总体“冒险程度”;
  2. top-p(nucleus)截断:只从累计概率 >= p 的高概率集合里采样,过滤长尾噪声;
  3. 重复惩罚:对最近若干步里出现过的符号,把它的 logit 压低,抑制复读机;
  4. 对 "/"(终止符)一视同仁采样,采到就自然停笔(配合上面抑制重复,很少死循环)。

用法:
  python generate.py                                          # v2 采样
  python generate.py --sampler naive                          # 和朴素采样对比
  python generate.py --model-kind onehot --model ../melody-rnn-lstm/model.keras
"""

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import music21 as m21

from preprocess import PROJECT_ROOT, SEQUENCE_LENGTH, TIME_STEP

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model.keras"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "mel.mid"


def load_model_and_mapping(model_path):
    model = keras.models.load_model(model_path)
    mapping = json.loads(
        (PROJECT_ROOT / "mapping.json").read_text(encoding="utf-8")
    )
    symbols = {index: symbol for symbol, index in mapping.items()}
    return model, mapping, symbols, len(mapping)


def make_input(window, vocab_size, kind):
    """把最近 SEQUENCE_LENGTH 个整数符号变成模型输入。"""
    window = np.asarray(window, dtype=np.int32)[np.newaxis, :]
    if kind == "int":
        return window
    return keras.utils.to_categorical(window, num_classes=vocab_size)


def _softmax(logits):
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def sample_naive(probabilities, temperature):
    """v1 式朴素采样:temperature 缩放 + 多项式采样(加 1e-8 防止 log(0) 出 NaN)。"""
    temperature = max(float(temperature), 1e-3)
    logits = np.log(probabilities + 1e-8) / temperature
    probs = _softmax(logits)
    return int(np.random.choice(len(probabilities), p=probs))


def sample_v2(probabilities, temperature, top_p, recent_tokens, repetition_penalty):
    """升级采样:重复惩罚 + top-p 截断 + 温度采样。"""
    temperature = max(float(temperature), 1e-3)
    logits = np.log(probabilities + 1e-8)

    if repetition_penalty > 0 and recent_tokens:
        # 对最近出现过的符号施加 logit 惩罚,降低它们再次被选中的概率
        for token in set(recent_tokens):
            logits[token] -= repetition_penalty

    logits = logits / temperature
    probs = _softmax(logits)

    # top-p / nucleus:保留“累计概率 >= top_p”的最小高概率集合
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    cumsum = np.cumsum(sorted_probs)
    keep = np.where(cumsum >= float(top_p))[0][0]  # 第一个越过阈值的位置
    keep = max(keep, 1)                            # 至少保留一个符号
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[: keep + 1]] = True
    probs = np.where(mask, probs, 0.0)
    probs = probs / probs.sum()

    return int(np.random.choice(len(probs), p=probs))


def generate_melody(model, mapping, symbols, seed_symbols, kind,
                    sampler, num_steps, temperature,
                    top_p=0.92, repetition_penalty=0.5, rep_window=8):
    """从种子续写,返回 (melody_symbols, stopped_by_delimiter)。"""
    melody = list(seed_symbols)
    int_seed = [mapping[s] for s in (["/"] * SEQUENCE_LENGTH) + seed_symbols]
    vocab_size = len(mapping)

    for _ in range(num_steps):
        window = int_seed[-SEQUENCE_LENGTH:]
        model_input = make_input(window, vocab_size, kind)
        probs = model.predict(model_input, verbose=0)[0]

        if sampler == "naive":
            next_index = sample_naive(probs, temperature)
        else:
            recent = int_seed[-rep_window:]
            next_index = sample_v2(
                probs, temperature, top_p, recent, repetition_penalty
            )

        next_symbol = symbols[next_index]
        if next_symbol == "/":
            return melody, True

        melody.append(next_symbol)
        int_seed.append(next_index)

    return melody, False


def save_melody(melody, step_duration=TIME_STEP, file_path=DEFAULT_OUTPUT_PATH):
    """把符号序列转成 music21 乐谱并写 MIDI。"""
    stream = m21.stream.Stream()
    current_symbol = None
    steps = 0

    def flush():
        nonlocal current_symbol, steps
        if current_symbol is None:
            return
        duration = step_duration * steps
        if current_symbol == "r":
            stream.append(m21.note.Rest(quarterLength=duration))
        else:
            stream.append(m21.note.Note(int(current_symbol), quarterLength=duration))
        current_symbol = None
        steps = 0

    for symbol in melody:
        if symbol == "_":
            steps += 1
        else:
            flush()
            current_symbol = symbol
            steps = 1
    flush()
    stream.write("midi", fp=str(file_path))
    print(f"MIDI 已保存到: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="v2 旋律生成")
    parser.add_argument("--seed", type=str,
                        default="67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--sampler", choices=["v2", "naive"], default="v2")
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=0.5)
    parser.add_argument("--rep-window", type=int, default=8)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--model-kind", choices=["int", "onehot"], default="int",
                        help="int=embedding 输入模型(v2);onehot=v1 模型")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    model, mapping, symbols, vocab_size = load_model_and_mapping(args.model)
    seed_symbols = args.seed.split()

    melody, stopped = generate_melody(
        model, mapping, symbols, seed_symbols,
        kind=args.model_kind,
        sampler="naive" if args.sampler == "naive" else "v2",
        num_steps=args.num_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        rep_window=args.rep_window,
    )
    print(f"生成完成,共 {len(melody)} 个符号;是否由 / 自然结束: {stopped}")
    print(" ".join(melody))
    save_melody(melody, file_path=args.output)


if __name__ == "__main__":
    main()