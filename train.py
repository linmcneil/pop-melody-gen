"""v2 训练:Embedding + 双层 LSTM + 验证集/早停/学习率衰减。

相对 v1(单层 LSTM + one-hot 输入)的“小创新”:
1. Embedding 代替逐时间步 one-hot:参数更省、训练更快,且不再需要把输入转成大稠密矩阵;
2. 堆叠两层 LSTM,让模型有更多非线性容量去学乐句级结构;
3. 划分验证集 + EarlyStopping + ReduceLROnPlateau:按验证损失挑最佳 epoch,
   缓解 v1 那种“不知道什么时候该停、容易过拟合训练集”的问题。

样本划分(为了和 v1 公平对比):
  用连续的样本区间:窗口 [0, train_samples) 训练,
  [train_samples, train_samples + val_samples) 验证(v1 训练时完全没见过这些窗口)。

运行:
  python train.py --epochs 40
输出: model.keras(以及训练过程)
"""

import argparse
from pathlib import Path

import keras
import tensorflow as tf

from preprocess import SEQUENCE_LENGTH, build_windows, load_mapping

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model.keras"

DEFAULT_EPOCHS = 40
DEFAULT_TRAIN_SAMPLES = 20000
DEFAULT_VAL_SAMPLES = 4000
DEFAULT_BATCH_SIZE = 64
DEFAULT_EMBED_DIM = 64
DEFAULT_LSTM_UNITS = [128, 128]
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_SEED = 42


def build_model(vocab_size, embed_dim, lstm_units, dropout, learning_rate):
    """Embedding -> 多层 LSTM(前几层 return_sequences=True)-> softmax。"""
    inputs = keras.Input(shape=(None,), dtype="int32")
    x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)

    for index, units in enumerate(lstm_units):
        return_sequences = index < len(lstm_units) - 1
        x = keras.layers.LSTM(units, return_sequences=return_sequences)(x)
        if return_sequences:
            x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def make_dataset(x_int, y_int, batch_size, shuffle_buffer=None, seed=DEFAULT_SEED):
    ds = tf.data.Dataset.from_tensor_slices((x_int, y_int))
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train(args):
    mapping = load_mapping()
    vocab_size = len(mapping)
    print(f"词表大小: {vocab_size}")

    # 一次性取出 train + val 的整数窗口,再按索引切成两段
    total_samples = args.train_samples + args.val_samples
    x_all, y_all = build_windows(SEQUENCE_LENGTH, max_samples=total_samples)
    x_train, y_train = x_all[: args.train_samples], y_all[: args.train_samples]
    x_val, y_val = x_all[args.train_samples:], y_all[args.train_samples:]
    print(f"训练样本: {len(x_train)} | 验证样本: {len(x_val)}")

    model = build_model(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        lstm_units=args.units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    model.summary()

    train_ds = make_dataset(
        x_train, y_train, args.batch_size,
        shuffle_buffer=min(10000, len(x_train)), seed=args.seed,
    )
    val_ds = make_dataset(x_val, y_val, args.batch_size)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2 if args.quiet else 1,
        shuffle=False,  # dataset 已洗牌
    )

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"v2 模型已保存到: {model_path}")


def parse_args():
    p = argparse.ArgumentParser(description="训练 v2 旋律生成模型")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    p.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    p.add_argument("--units", type=str, default=",".join(map(str, DEFAULT_LSTM_UNITS)),
                   help="LSTM 各层单元数,逗号分隔")
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.units = [int(u) for u in args.units.split(",") if u.strip()]
    train(args)