# -*- coding: utf-8 -*-
"""models_v3: 同一组超参下的三种自回归模型, 供公平对比。

base       单流 token 输入的 Embedding+双层 LSTM(v2 架构的复刻, 修复数据划分后重训);
cond       双流输入: token + 逐 step 和弦符号, 和弦 Embedding 与旋律 Embedding 拼接进 LSTM;
xformer    小尺寸因果 Transformer(2 层/4 头/128 维), 作为更强的长程基线。
"""
import numpy as np
import keras
import tensorflow as tf


def compile_model(model, lr=1e-3):
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=["accuracy"],
    )
    return model


def build_lstm_base(vocab_size, embed_dim=64, lstm_units=(128, 128), dropout=0.2):
    inputs = keras.Input(shape=(None,), dtype="int32")
    x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)
    for i, units in enumerate(lstm_units):
        rs = i < len(lstm_units) - 1
        x = keras.layers.LSTM(units, return_sequences=rs)(x)
        if rs:
            x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(vocab_size, activation="softmax")(x)
    return compile_model(keras.Model(inputs, out))


def build_lstm_cond(vocab_size, chord_vocab_size, embed_dim=64, chord_embed_dim=24,
                    lstm_units=(128, 128), dropout=0.2):
    mel_in = keras.Input(shape=(None,), dtype="int32", name="melody")
    chd_in = keras.Input(shape=(None,), dtype="int32", name="chord")
    m = keras.layers.Embedding(vocab_size, embed_dim)(mel_in)
    c = keras.layers.Embedding(chord_vocab_size, chord_embed_dim)(chd_in)
    x = keras.layers.Concatenate()([m, c])      # 每步: 旋律符号 + 该步所属和弦
    for i, units in enumerate(lstm_units):
        rs = i < len(lstm_units) - 1
        x = keras.layers.LSTM(units, return_sequences=rs)(x)
        if rs:
            x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(vocab_size, activation="softmax")(x)
    return compile_model(keras.Model([mel_in, chd_in], out))


@keras.saving.register_keras_serializable()
class TransformerLM(keras.Model):
    """极简因果 Transformer: Embedding + 学习位置 + 2 层 MHA, 逐 token softmax。"""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 ff_dim=256, dropout=0.1, lr=3e-4, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.lr = lr
        self.embed = keras.layers.Embedding(vocab_size, d_model)
        self.pos_embed = self.add_weight(
            name="pos", shape=(1, 512, d_model), initializer="normal", trainable=True)
        self.drop = keras.layers.Dropout(dropout)
        self.blocks = [
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ]
        self.ln = keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = keras.layers.Dense(vocab_size, activation="softmax")
        self.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy"],
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size, "d_model": self.d_model,
            "n_heads": self.n_heads, "n_layers": self.n_layers,
            "ff_dim": self.ff_dim, "dropout": self.dropout, "lr": self.lr,
        })
        return config

    def call(self, x, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        h = self.embed(x) * tf.sqrt(tf.cast(self.d_model, tf.float32))
        h = h + self.pos_embed[:, :T, :]
        h = self.drop(h, training=training)
        for block in self.blocks:
            h = block(h, training=training)
        h = self.ln(h)
        # 与 LSTM 的任务一致: 用整窗 64 步上下文, 只预测窗口后的下一个符号
        h = h[:, -1, :]
        return self.head(h)


@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, n_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.attn = keras.layers.MultiHeadAttention(n_heads, d_model // n_heads,
                                                    output_shape=d_model, dropout=dropout)
        self.ffn1 = keras.layers.Dense(ff_dim, activation="relu")
        self.ffn2 = keras.layers.Dense(d_model)
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.drop = keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads,
                       "ff_dim": self.ff_dim, "dropout": self.dropout})
        return config

    def call(self, x, training=False):
        T = tf.shape(x)[1]
        mask = tf.linalg.band_part(tf.ones((1, T, T), dtype=tf.bool), -1, 0)
        a = self.attn(x, x, attention_mask=mask, training=training)
        x = self.ln1(x + self.drop(a, training=training))
        f = self.ffn2(self.ffn1(x))
        return self.ln2(x + self.drop(f, training=training))
