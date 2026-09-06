# -*- coding: utf-8 -*-
"""make_figs.py - 生成 README/报告用图: 全流程管线 + 和弦条件 LSTM 结构。
运行: python scripts/make_figs.py  (输出 docs/pipeline.png, docs/model_arch.png)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"):
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

C1 = "#DCE9F7"; E1 = "#2F6FB2"   # data blue
C2 = "#EDE3F7"; E2 = "#6A3FA0"   # train purple
C3 = "#E2F2E5"; E3 = "#2E8B57"   # eval green
C4 = "#FDEBD0"; E4 = "#D97904"   # output orange


def box(ax, x, y, w, h, fc, ec, lw=1.6, r=0.8):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.25,rounding_size={r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    return b


def arrow(ax, x1, y1, x2, y2, color="#444444"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22,
                        linewidth=1.8, color=color, zorder=1)
    ax.add_patch(a)


def chip(ax, x, y, w, h, text, ec=E2, fs=10):
    box(ax, x, y, w, h, "#FFFFFF", ec, lw=1.2, r=0.5)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# =============================== 1) 全流程管线
fig, ax = new_fig(1500, 850)
ax.text(750, 822, "Pop Melody Gen - 流行主旋律自动生成全流程", ha="center",
        fontsize=25, fontweight="bold", color="#1F2A44")

ax.text(16, 742, "1) 数据 / Data", fontsize=16, fontweight="bold", color=E1)
box(ax, 16, 600, 1468, 124, C1, E1, lw=2)
ax.text(32, 688, "输入: Pop-K 300k 条 8 小节现代流行 loop(MIDI)", fontsize=14, color="#111")
ax.text(32, 660, "单条音轨 = 低音 + 和弦 + 主旋律叠在一起, 并没有现成的干净人声主旋律轨", fontsize=12.5, color="#333")
ax.text(32, 638, "处理: 主旋律层抽取 -> 0.25 拍网格量化 -> Krumhansl 移调 / 八度归一 -> 由伴奏推逐小节和弦", fontsize=12.5, color="#333")
ax.text(32, 614, "输出: 7,000 条干净单声部主旋律 + 逐 token 和弦标签(按歌划分 5600 / 700 / 700)", fontsize=12.5, color=E1, fontweight="bold")
arrow(ax, 750, 598, 750, 534)

ax.text(16, 500, "2) 建模 / Training", fontsize=16, fontweight="bold", color=E2)
box(ax, 16, 320, 1468, 170, C2, E2, lw=2)
ax.text(32, 456, "任务: 给定最近 64 个时间步(每步 0.25 拍)预测下一符号; 词表 49(音高 / 延音 _ / 休止 r / 分隔 /)", fontsize=12.5, color="#333")
ax.text(32, 436, "对比模型(CPU 训练, 每 epoch 随机抽 2 万窗口, 30 epoch 上限, 固定验证/测试窗口共用):", fontsize=12.5, color="#333")
cw, chh, gap = 218, 58, 14
chips = ["Markov-6", "LSTM\n(单流)", "LSTM+和弦\n(双流)", "LSTM\n(重平衡 6x)", "LSTM+和弦\n(重平衡 6x)", "Transformer\n(2 层)"]
x = 30
for lab in chips:
    chip(ax, x, 344, cw, chh, lab, fs=13)
    x += cw + gap
ax.text(32, 356, "", fontsize=1)
arrow(ax, 750, 318, 750, 256)

ax.text(16, 224, "3) 评测 / Evaluation", fontsize=16, fontweight="bold", color=E3)
box(ax, 16, 46, 1468, 170, C3, E3, lw=2)
ax.text(32, 186, "客观指标(固定 4,000 条 test 窗口; 全部按无权重 CE 上报):", fontsize=13, color="#111", fontweight="bold")
ax.text(32, 158, "- test NLL 0.783~0.943 / PPL 2.19~2.57; Transformer 2.19 < LSTM 2.25 < Markov 2.50", fontsize=12.5, color="#333")
ax.text(32, 134, "- 音符启动(note-onset)top-1: vanilla CE 约 3%(几乎随机, 只会输出延音); 重平衡后升到 28%~30%", fontsize=12.5, color="#333")
ax.text(32, 110, "- 生成质量(12 条 test prompt, temp 0.7, C-Am-F-G): 和弦版平均续写 76 tokens / 426 音符; 纯旋律版 25 tokens / 135 音符", fontsize=12.5, color="#333")
ax.text(32, 84, "人类听感: A/B/C 三条件材料 + Friedman / Wilcoxon 协议(见 human_eval/)", fontsize=12.5, color="#333")
ax.text(32, 58, "交付物: v3-demo-cond-*.mid/.wav(3 套进行) + v3-demo-base.* + REPORT_TABLE.md + HIGHLIGHTS.md", fontsize=12.5, color=E4, fontweight="bold")
fig.savefig(DOCS / "pipeline.png", dpi=120, bbox_inches="tight", facecolor="white")
plt.close(fig)

# =============================== 2) 双流 LSTM 结构
fig, ax = new_fig(1360, 540)
ax.text(680, 505, "Chord-conditioned LSTM - 双流输入结构", ha="center",
        fontsize=20, fontweight="bold", color="#1F2A44")


def io_box(x, y, w, h, lines, fc="#FFFFFF", ec="#444444"):
    box(ax, x, y, w, h, fc, ec, lw=1.5, r=0.6)
    ax.text(x + w / 2, y + h / 2 + 14, lines[0], ha="center", va="center",
            fontsize=14, color="#111", zorder=3)
    for k, ln in enumerate(lines[1:], start=1):
        ax.text(x + w / 2, y + h / 2 - 14 * k, ln, ha="center", va="center",
                fontsize=11, color="#555", zorder=3)


emb_c = "#FDF3D8"; emb_e = "#B07A00"
lstm_c = "#E8F0FE"; lstm_e = "#1E5FAA"

io_box(24, 316, 220, 140, ["旋律 token", "x_1 ... x_64", "(49 类, 每步 0.25 拍)"])
arrow(ax, 244, 386, 306, 386)
box(ax, 306, 326, 150, 120, emb_c, emb_e, lw=1.6)
ax.text(381, 376, "Embedding", ha="center", va="center", fontsize=13, zorder=3)
ax.text(381, 352, "64 维", ha="center", va="center", fontsize=11, color="#7a5a00", zorder=3)

io_box(24, 70, 220, 140, ["和弦标签", "c_1 ... c_64", "(25 类, 逐 token 对齐)"])
arrow(ax, 244, 140, 306, 140)
box(ax, 306, 80, 150, 120, emb_c, emb_e, lw=1.6)
ax.text(381, 130, "Embedding", ha="center", va="center", fontsize=13, zorder=3)
ax.text(381, 106, "24 维", ha="center", va="center", fontsize=11, color="#7a5a00", zorder=3)

box(ax, 520, 140, 128, 220, "#F3E8FA", "#7B3FA0", lw=1.8)
ax.text(584, 318, "Concatenate", ha="center", va="center", fontsize=13, zorder=3)
ax.text(584, 200, "88 维/步", ha="center", va="center", fontsize=11.5, color="#4b2a66", zorder=3)
ax.text(584, 176, "旋律语义 + 和声语义", ha="center", va="center", fontsize=10.5, color="#4b2a66", zorder=3)
arrow(ax, 456, 386, 566, 320)
arrow(ax, 456, 140, 566, 240)

box(ax, 700, 120, 230, 250, lstm_c, lstm_e, lw=2)
ax.text(815, 330, "LSTM x2", ha="center", va="center", fontsize=16, fontweight="bold", color=lstm_e, zorder=3)
ax.text(815, 300, "hidden 128", ha="center", va="center", fontsize=11.5, color="#1c3f6e", zorder=3)
ax.text(815, 276, "双流拼接输入", ha="center", va="center", fontsize=11, color="#1c3f6e", zorder=3)
ax.text(815, 252, "(CPU 可训, 约 0.2-0.3M 参数)", ha="center", va="center", fontsize=10.5, color="#1c3f6e", zorder=3)
arrow(ax, 648, 245, 700, 245)

box(ax, 990, 140, 176, 190, "#FDE9E0", "#C0392B", lw=1.8)
ax.text(1078, 290, "Dense + softmax", ha="center", va="center", fontsize=13, zorder=3)
ax.text(1078, 190, "下一个符号", ha="center", va="center", fontsize=12.5, fontweight="bold", zorder=3)
ax.text(1078, 166, "(49 类)", ha="center", va="center", fontsize=11, color="#7a2a1a", zorder=3)
arrow(ax, 930, 245, 990, 245)

ax.text(24, 42, "训练时: c_1..c_64 = 该旋律自己推断出的逐小节和弦;  生成(解码)时: 由用户指定进行(如 C-Am-F-G), 逐小节重复喂入 -> 可控生成。",
        fontsize=12, color="#555")
fig.savefig(DOCS / "model_arch.png", dpi=120, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("written:", DOCS / "pipeline.png", DOCS / "model_arch.png")