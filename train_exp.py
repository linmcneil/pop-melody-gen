# -*- coding: utf-8 -*-
"""train_exp: 在 dataset_v3 上训练一个实验模型并输出指标。

用法:
  python train_exp.py --kind base   --name base_s1 --seed 1
  python train_exp.py --kind cond   --name cond_s1 --seed 1
  python train_exp.py --kind xformer --name xf_s1 --seed 1

每个 epoch 从整个 train split 里重新随机抽 n_train 窗口; 验证窗口固定缓存。
早停后恢复最佳权重, 在固定测试窗口上报 loss/acc 并存 results.json。
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

import corpus as C
import models_v3 as M

PROJECT_ROOT = Path(__file__).resolve().parent
EXP_DIR = PROJECT_ROOT / "experiments"

SEQ_LEN = C.SEQ_LEN
EPOCHS_DEFAULT = 30
TRAIN_W_DEFAULT = 20000
VAL_W_DEFAULT = 2000
TEST_W_DEFAULT = 4000
BATCH = 64
PATIENCE = 4
LR_PATIENCE = 2


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def build_model(kind, vocab_size, chord_vocab_size):
    if kind == "base":
        return M.build_lstm_base(vocab_size)
    if kind == "cond":
        return M.build_lstm_cond(vocab_size, chord_vocab_size)
    if kind == "xformer":
        return M.TransformerLM(vocab_size)
    raise ValueError(kind)


def run(args):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    co = C.Corpus()

    xt, yt = None, None
    note_ids = np.asarray([i for i, sym in enumerate(co.symbol_vocab)
                           if sym not in ("_", "r", "/")], dtype=np.int32)
    def train_weights(y):
        if args.note_weight <= 1.0:
            return None
        return np.where(np.isin(y, note_ids), args.note_weight, 1.0).astype(np.float32)

    xv, cv, yv = co.fixed_windows("val", VAL_W_DEFAULT)
    xte, cte, yte = co.fixed_windows("test", TEST_W_DEFAULT)
    wv = train_weights(yv)
    val_ds = C.tf_windows(xv, yv, BATCH, cv if args.kind == "cond" else None, wv)

    model = build_model(args.kind, co.n_sym, co.n_chord)
    print(f"[{args.name}] kind={args.kind} vocab={co.n_sym} chord_vocab={co.n_chord} seed={args.seed}")
    model.summary()

    best_val, best_ep, no_impr, lr_no_impr = 1e9, -1, 0, 0
    opt = model.optimizer
    # 类别重平衡(见上方 train_weights): 音符启动 token 在 vanilla CE 下会被模型
    # “战略性忽略”(只输出 _ 和 /), 权重>1 逼迫模型学会再攻击。验证集用同一目标选早停。
    for ep in range(1, args.epochs + 1):
        x, c, y = co.train_windows(args.train_windows, seed=args.seed * 1000 + ep)
        w = train_weights(y)
        ds = C.tf_windows(x, y, BATCH, c if args.kind == "cond" else None, w)
        hist = model.fit(ds, epochs=1, verbose=1, shuffle=False)
        vloss, vacc = model.evaluate(val_ds, verbose=0)
        print(f"  epoch {ep:02d}: train_loss={hist.history['loss'][0]:.4f} "
              f"val_loss={vloss:.4f} val_acc={vacc:.4f}")
        if vloss < best_val - 1e-4:
            best_val, best_ep = vloss, ep
            no_impr, lr_no_impr = 0, 0
            model.save_weights(str(EXP_DIR / f"{args.name}_best.weights.h5"))
        else:
            no_impr += 1
            lr_no_impr += 1
            if lr_no_impr >= LR_PATIENCE and lr_no_impr < PATIENCE:
                lr = float(opt.learning_rate.numpy()) * 0.5
                opt.learning_rate.assign(max(lr, 1e-6))
                print(f"    reduce lr -> {lr:.2e}")
            if no_impr >= PATIENCE:
                print(f"early stop at epoch {ep}, best val at {best_ep} ({best_val:.4f})")
                break

    model.load_weights(str(EXP_DIR / f"{args.name}_best.weights.h5"))
    test_ds = C.tf_windows(xte, yte, BATCH, cte if args.kind == "cond" else None)
    tloss, tacc = model.evaluate(test_ds, verbose=0)
    print(f"test loss={tloss:.4f} acc={tacc:.4f}")
    model.save(str(EXP_DIR / f"{args.name}.keras"))
    print(f"saved {EXP_DIR / (args.name + '.keras')}")

    result = {
        "name": args.name, "kind": args.kind, "seed": args.seed,
        "best_epoch": best_ep, "best_val_loss": float(best_val),
        "test_loss": float(tloss), "test_acc": float(tacc),
        "n_train_windows_per_epoch": args.train_windows,
        "note_weight": args.note_weight,
        "epochs_used": best_ep,
        "n_val_windows": VAL_W_DEFAULT, "n_test_windows": TEST_W_DEFAULT,
    }
    (EXP_DIR / f"{args.name}_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    # 清掉中间权重文件
    (EXP_DIR / f"{args.name}_best.weights.h5").unlink(missing_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["base", "cond", "xformer"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--train-windows", type=int, default=TRAIN_W_DEFAULT)
    p.add_argument("--note-weight", type=float, default=1.0,
                   help="音符启动 token 的样本权重(>1 启用类别重平衡)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
