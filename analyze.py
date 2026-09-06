# -*- coding: utf-8 -*-
"""analyze.py — 在固定 test 窗口上按 GT 类别拆分统计每个模型的预测能力。

动机: vanilla CE 的模型会用“永远输出 _/”这种懒惰解把整体 acc 刷到 0.75,
但音符启动(note-onset)的 top-1 准确率可能低到 ~3%(接近随机)。本脚本把
overall / note / hold / rest / sep 分开报, 让“重平衡是否真的让模型会出音”
可以被直接量化。

用法:
  python analyze.py
输出: experiments/note_metrics.json + 控制台表格
"""
import json
import math
from pathlib import Path

import numpy as np
import keras

import corpus as C
import models_v3  # noqa: F401 (注册 TransformerLM 以便 .keras 反序列化)

PROJECT_ROOT = Path(__file__).resolve().parent
EXP_DIR = PROJECT_ROOT / "experiments"

CATS = {"note": ("_not_a_real_tag_", None)}


def _cat_of(sym):
    if sym in ("_", "r", "/"):
        return {"_": "hold", "r": "rest", "/": "sep"}[sym]
    return "note"


def run_all():
    co = C.Corpus()
    x, c, y = co.fixed_windows("test", 4000)
    i2s = co.symbol_vocab
    cats = np.asarray([_cat_of(i2s[t]) for t in y])
    out = []
    for p in sorted(EXP_DIR.glob("*_results.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        name, kind = r["name"], r["kind"]
        if kind not in ("base", "cond", "xformer"):
            continue
        mpath = EXP_DIR / f"{name}.keras"
        if not mpath.exists():
            continue
        from keras.saving import custom_object_scope
        if kind == "xformer":
            with custom_object_scope({"TransformerLM": models_v3.TransformerLM,
                                      "TransformerBlock": models_v3.TransformerBlock}):
                model = keras.models.load_model(str(mpath))
        else:
            model = keras.models.load_model(str(mpath))
        if kind == "cond":
            pred = model.predict([x, c], verbose=0, batch_size=256)
        else:
            pred = model.predict(x, verbose=0, batch_size=256)
        arg = np.argmax(pred, axis=-1)
        row = {"name": name, "kind": kind, "note_weight": r.get("note_weight", 1.0),
               "test_loss": r.get("test_loss"), "test_acc": float((arg == y).mean())}
        for cat in ("note", "hold", "rest", "sep"):
            m = cats == cat
            if m.sum() == 0:
                row[f"acc_{cat}"] = None
                continue
            row[f"acc_{cat}"] = float((arg[m] == y[m]).mean())
            row[f"n_{cat}"] = int(m.sum())
        out.append(row)
        print(f"{name} (kind={kind}, note_weight={row['note_weight']}): "
              f"acc={row['test_acc']:.4f} note={row['acc_note']:.3f} "
              f"hold={row['acc_hold']:.3f} rest={row['acc_rest']} sep={row['acc_sep']:.3f}")
    (EXP_DIR / "note_metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written", EXP_DIR / "note_metrics.json")


if __name__ == "__main__":
    run_all()