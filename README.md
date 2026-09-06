# Pop Melody Gen — 流行音乐主旋律自动生成(Chord-conditioned LSTM)

一个从“复刻经典 LSTM 旋律生成教程([musikalkemist/generating-melodies-with-rnn-lstm](https://github.com/musikalkemist/generating-melodies-with-rnn-lstm))”出发、最终收敛到 **现代流行主旋律 + 和弦可控生成** 的小型研究型项目。
全部在 **Windows + CPU** 上可复现(单 epoch ~35–45s,整轮训练 15–20 分钟)。

![pipeline](docs/pipeline.png)

## 这个项目解决什么

- 目标:**给定和弦进行(例如 C–Am–F–G),让模型写一段流行主旋律**,或给一段开头让它接着写。
- 难点:公开“流行 MIDI”大多是整段伴奏 loop,没有干净的“人声主旋律轨”;而且把旋律编码成 0.25 拍网格后,
  **vanilla 交叉熵会学会“只输出延音/结束符、从不发起新音符”的懒惰解**(本项目实测 note-onset top-1 只有 ~3%)。
- 我们的做法:先做旋律层抽取 + 逐小节和弦推断,再做**音符启动样本重平衡训练**,最后对比
  Markov-6 / 单流 LSTM / 双流“旋律+和弦”LSTM / 2 层 Transformer,并用固定窗口 + 按歌划分保证公平。

## 目录结构(与 v3 主线相关)

```
pop-melody-gen/
├─ data/                          # 原始语料放这里(Git 忽略;用 download_dataset.py 获取)
├─ preprocess_v3.py               # 旋律抽取 + 和弦推断 → dataset_v3/ (7000 条 + 逐 token 和弦)
├─ dataset_v3/                    # <id>_tokens.txt / <id>_chords.txt / splits.json / stats.json
├─ corpus.py                      # 按歌 8:1:1 划分;训练随机采样窗口;固定验证/测试窗口;Markov 基线
├─ models_v3.py                   # base LSTM / chord-conditioned LSTM / TransformerLM(Keras 3)
├─ train_exp.py                   # 实验训练脚本(--note-weight 支持类别重平衡)
├─ analyze.py                     # 按 GT 类别(note/hold/rest/sep)拆分的预测准确率
├─ eval_v3.py                     # 统一评测 → REPORT_TABLE.md + experiments/summary.json
├─ generate_v3.py                 # 采样(温度+top-p+重复惩罚)→ MIDI + 轻量 WAV
├─ human_eval/                    # A/B/C 人类听感材料生成与分析(Friedman/Wilcoxon)
├─ experiments/                   # 预训练模型 *.keras + *_results.json(已随仓库发布)
├─ docs/                          # 流程图 pipeline.png / model_arch.png(用 scripts/make_figs.py 重绘)
├─ REPORT_TABLE.md                # 最终数字表
├─ v3-demo-*.mid / *.wav          # 可直接试听的成品(cond 三套进行 + base 对照)
└─ HIGHLIGHTS.md / REPORT.md      # 一页亮点 / 完整研究主线报告
```


## 预训练模型(仓库内直接可用)

五个训练好的模型已随仓库提交到 `experiments/`,可直接加载评测/生成,不必重训:

| 文件 | 说明 | test NLL |
|---|---|---|
| `xf_s1.keras` | Transformer-2L(vanilla CE) | 0.783 |
| `base_s1.keras` | 单流 LSTM(vanilla CE) | 0.809 |
| `cond_s1.keras` | 双流 LSTM+和弦(vanilla CE) | 0.824 |
| `base_r1.keras` | 单流 LSTM(音符重平衡 6x) | 0.940 |
| `cond_r1.keras` | 双流 LSTM+和弦(音符重平衡 6x,主力 demo 模型) | 0.943 |

> 注: vanilla CE 模型 NLL 更低但采样“几乎不出新音”;重平衡模型(带 `_r1`)虽然无权重
> NLL 略高,才是能实际“写旋律”的版本。直接出 demo:

```powershell
& $PY generate_v3.py --base-model base_r1 --cond-model cond_r1 --temperature 0.7
```

生成/评测脚本(`generate_v3.py`、`eval_v3.py`、`analyze.py`)默认会读取这些文件;
Transformer 由 `models_v3.TransformerLM` 反序列化(脚本内已处理)。

双流模型的输入结构示意:

![model_arch](docs/model_arch.png)

## 快速复现(v3 主线)

```powershell
$PY = "C:\Users\lzq13\melody-rnn-lstm\.venv\Scripts\python.exe"

# 0) 下载 Pop-K 原始语料(约 54 MB, Zenodo, CC BY-NC)——发布版仓库不携带 data/
& $PY scripts\download_dataset.py

# 1) 数据:旋律抽取 + 和弦推断 + 按歌划分(首次需解压,约几分钟)
& $PY preprocess_v3.py

# 2) 训练(同一套固定验证/测试窗口,种子=1)
& $PY train_exp.py --kind base    --name base_s1  --seed 1            # vanilla CE
& $PY train_exp.py --kind cond    --name cond_s1  --seed 1
& $PY train_exp.py --kind base    --name base_r1  --seed 1 --note-weight 6   # 重平衡
& $PY train_exp.py --kind cond    --name cond_r1  --seed 1 --note-weight 6
& $PY train_exp.py --kind xformer --name xf_s1    --seed 1

# 3) 评测:按类准确率 + 统一生成质量表
& $PY analyze.py
& $PY eval_v3.py --num-seeds 12 --num-steps 160 --temperature 0.7

# 4) 生成 demo MIDI/WAV
& $PY generate_v3.py --base-model base_r1 --cond-model cond_r1 --temperature 0.7

# 5) 人类听感 A/B/C 材料(可选)
& $PY human_eval\make_stimuli.py --base base_r1 --cond cond_r1 --n-loops 4
```

## 最终结果速览(详见 REPORT_TABLE.md / HIGHLIGHTS.md)

- 测试 NLL:Transformer 0.783 < 普通 LSTM 0.809 < Markov-6 0.916(PPL 2.19 / 2.25 / 2.50)。
- vanilla CE 的 note-onset top-1 准确率仅 2–3%;加音符重平衡后 base 28%、**和弦版 30%**,采样能正常“发起新音”。
- 在 12 条 test prompt + C–Am–F–G 下:**和弦版重平衡模型平均续写 76 tokens、合计 426 个音符**,
  而纯旋律版只有 25 tokens / 135 个音符——**给定和声,模型能连续写得更长、更密、音域更宽**(range 9.9 vs 6.2)。
- 真实人声主旋律的音符-和弦贴合率约 36.8%(随机音高 25%),生成旋律在小调/大调音阶内保持贴和;听感材料见
  `human_eval/stimuli/`。

## 历史

- v1/v2 在德国民歌语料(KERN)上复刻并改进教程模型(naive → 温度+top-p+重复惩罚),验证了“采样器改进在换域后仍成立”;
- v3 换到流行语料 Pop-K,自己写“整轨 loop → 干净主旋律 + 和弦标注”的数据管线,并按上面协议做正式对比。
- 完整的研究主线、动机、困难与解决,见 `REPORT.md`;CV 用的一页亮点见 `HIGHLIGHTS.md`;英文版快速说明见 `README.en.md`。

## 参考与致谢 (Acknowledgements)

本项目以 Valerio Velardo 的 [musikalkemist/generating-melodies-with-rnn-lstm](https://github.com/musikalkemist/generating-melodies-with-rnn-lstm) 教程
(Generating Melodies with RNN-LSTM, MIT License) 为学习起点：
先完整复刻其数据与 LSTM 训练流程，
再逐步加入采样改进、双流和弦条件模型与
自建流行语料管线，最终形成本仓库 v3 主线。
感谢原作者与 The Sound of AI 社区。
