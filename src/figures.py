"""발표·보고서용 그림. summary 폴더의 집계 CSV만 읽어서 그린다(모델을 다시 돌리지 않음).

    python -m src.analyze --exp results/6conv_fold     # 먼저 집계
    python -m src.figures --exp results/6conv_fold     # -> <exp>/figures/*.pdf, *.png

그림 목록
- fig1_bits:    비트 수에 따른 정답률, 4비트에서 W와 A 중 어느 쪽 영향이 큰지
- fig2_layers:  W4A4 층별 누적 NMSE / 복원 이득 / 단독 하락, 그리고 두 순위의 관계
- fig3_pareto:  Mixed-Precision 64조합의 저장 비용 대 정답률과 Pareto frontier
- fig4_domain:  입력 조건 x 정밀도 조건 정답률 표(heatmap)
- fig5_gain3d:  입력 조건 x 층별 복원 이득 (3D 막대)

그리는 원칙
- 축은 하나만 쓴다. 단위가 다른 값(NMSE, %p)은 한 축에 겹치지 않고 칸을 나눈다.
- seed 편차가 커서 평균만 그리지 않고 seed별 점을 함께 찍는다.
- 글자는 영어. 한글 글꼴이 없는 환경에서도 깨지지 않게 하려는 것이다.
- fig5의 3D 막대는 음수를 그릴 수 없어 음수 복원 이득(Conv6, 최대 -0.7%p)을 0으로 그린다.
  정확한 값은 fig2나 domain_layers.csv를 볼 것.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # 색각 이상에서도 구분되는지 검증한 조합
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"])  # 한 색 밝기 단계
AVG = "mean3"
LAYER_NAMES = [f"Conv{i}" for i in range(1, 7)]
DOMAIN_NAMES = {"orig": "Original", "bg": "Background\nswap", "noise_0.01": "Noise\nσ=0.01",
                "noise_0.03": "Noise\nσ=0.03", "blur_0.5": "Blur\nσ=0.5", "blur_1.0": "Blur\nσ=1.0"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": True, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", color=INK)


def seed_dots(ax, x, values, color=MUTED) -> None:
    """seed별 값을 작은 점으로. 좌우로 조금 흩어서 겹침을 줄인다."""
    offsets = np.linspace(-0.16, 0.16, len(values))
    ax.scatter(x + offsets, values, s=9, color=color, edgecolor="white", linewidth=0.4, zorder=3)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(out / f"{name}.png")


def fig1_bits(summary: Path, out: Path) -> None:
    cond = pd.read_csv(summary / "conditions.csv")
    acc = cond[cond.task == AVG].pivot_table(index="seed", columns="condition", values="acc")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})

    order = ["W4A4", "W5A5", "W6A6", "W8A8", "FP32"]
    labels = ["4", "5", "6", "8", "FP32"]
    x = np.arange(len(order))
    for seed, row in acc[order].iterrows():
        a.plot(x, row.to_numpy(), color=GRID, linewidth=1, zorder=1)  # seed 하나의 궤적
        a.scatter(x, row.to_numpy(), s=9, color=MUTED, edgecolor="white", linewidth=0.4, zorder=2)
    mean = acc[order].mean()
    a.plot(x, mean.to_numpy(), color=BLUE, linewidth=2, marker="o", markersize=5, zorder=3, label="mean of 5 seeds")
    a.scatter([], [], s=9, color=MUTED, label="single seed")
    for xi, v, offset in zip(x[:2], mean[:2], ((8, -3), (5, -11))):
        a.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=offset, color=INK, fontsize=7.5)
    a.set_xticks(x, labels)
    a.set_xlabel("bit-width (weights = activations)")
    a.set_ylabel("top-1 accuracy (%)")
    a.set_ylim(35, 103)
    a.legend(loc="lower right")
    panel(a, "a", "Accuracy collapses between 5 and 4 bits")

    order = ["W4A32", "W32A4", "W4A4"]
    labels = ["weights only\n(W4 A32)", "activations only\n(W32 A4)", "both\n(W4 A4)"]
    x = np.arange(len(order))
    b.bar(x, acc[order].mean(), width=0.5, color=BLUE, zorder=2)
    for xi, c in zip(x, order):
        seed_dots(b, xi, acc[c].to_numpy(), INK)
        b.text(xi, 103, f"{acc[c].mean():.1f}", ha="center", va="bottom", color=INK, fontsize=7.5)
    b.set_xticks(x, labels)
    b.set_ylim(0, 112)
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_ylabel("top-1 accuracy (%)")
    panel(b, "b", "What is quantized to 4 bits")
    fig.tight_layout(w_pad=2)
    save(fig, out, "fig1_bits")


def fig2_layers(summary: Path, out: Path) -> None:
    layers = pd.read_csv(summary / "layers.csv")
    w = layers[(layers.condition == "W4A4") & (layers.task == AVG)]
    corr = pd.read_csv(summary / "rank_corr.csv")
    rho = corr[(corr.condition == "W4A4") & (corr.task == AVG) & (corr.x == "nmse_all") & (corr.y == "gain_pp")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    x = np.arange(6)
    specs = [("nmse_all", "Cumulative output error", "NMSE", MUTED),
             ("gain_pp", "Gain from restoring to FP32", "accuracy gain (%p)", BLUE),
             ("drop_pp", "Drop from quantizing alone", "accuracy drop (%p)", BLUE)]
    for ax, letter, (col, title, ylabel, color) in zip(axes, "abc", specs):
        wide = w.pivot_table(index="seed", columns="layer", values=col)
        ax.bar(x, wide.mean(), width=0.62, color=color, zorder=2)
        for i in x:
            seed_dots(ax, i, wide[i + 1].to_numpy(), INK)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_xticks(x, [str(i) for i in range(1, 7)])
        ax.set_xlabel("conv layer")
        ax.set_ylabel(ylabel)
        panel(ax, letter, title)

    # (d) 두 순위를 직접 비교. 1위가 가장 큰 값. 대각선 위에 있으면 두 기준이 같은 층을 고른 것이다
    ax = axes[3]
    ax.plot([1, 6], [1, 6], color=GRID, linewidth=1, zorder=1)
    # 같은 칸에 여러 seed가 겹치므로 칸 안에서 작은 원 모양으로 흩는다
    angles = np.linspace(0, 2 * np.pi, w.seed.nunique(), endpoint=False)
    for angle, (seed, g) in zip(angles, w.groupby("seed")):
        g = g.sort_values("layer")
        rx = g.nmse_all.rank(ascending=False).to_numpy() + 0.2 * np.cos(angle)
        ry = g.gain_pp.rank(ascending=False).to_numpy() + 0.2 * np.sin(angle)
        ax.scatter(rx, ry, s=14, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(range(1, 7))
    ax.set_yticks(range(1, 7))
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.5, 6.5)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("rank by cumulative error")
    ax.set_ylabel("rank by restoration gain")
    sig = int((rho.rho.abs() >= 0.886).sum())
    ax.set_xlabel("rank by cumulative error")
    # 유의한 seed 수(|ρ| ≥ 0.886)는 그림 설명에 적는다. 현재 5개 중 sig개
    panel(ax, "d", f"Ranks disagree (mean ρ = {rho.rho.mean():.2f})".replace("-", "\u2212"))
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "fig2_layers")


def fig3_pareto(summary: Path, out: Path) -> None:
    mix = pd.read_csv(summary / "mixed_all.csv", dtype={"bits": str})
    mix = mix[mix.task == AVG].groupby(["bits", "weight_bit_cost"], as_index=False).acc.mean()
    front = pd.read_csv(summary / "pareto.csv", dtype={"bits": str})
    front = front[front.scope == "mean"].sort_values("weight_bit_cost")
    last8 = mix.bits.str[-1] == "8"  # 누적 NMSE가 가장 큰 Conv6에 8비트를 쓴 조합
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.grid(axis="x")
    ax.scatter(mix.weight_bit_cost[~last8], mix.acc[~last8], s=20, color=BLUE, edgecolor="white", linewidth=0.6,
               zorder=3, label="Conv6 kept at 4 bits")
    ax.scatter(mix.weight_bit_cost[last8], mix.acc[last8], s=20, color=ORANGE, marker="s", edgecolor="white",
               linewidth=0.6, zorder=3, label="Conv6 raised to 8 bits")
    ax.step(front.weight_bit_cost, front.test_acc, where="post", color=INK, linewidth=1.2, zorder=2,
            label="Pareto frontier (chosen on rank set)")
    # frontier 조합 이름. 점이 몰린 곳은 이름표를 빈 곳에 두고 가는 선으로 잇는다.
    # 값은 (가로, 세로) 오프셋(pt). 세 번째 값이 True면 선을 그린다
    place = {"444444": (8, -3, False), "844444": (8, -3, False), "884444": (-8, 3, False),
             "888444": (-8, 3, False), "884844": (-38, 20, True), "888844": (-4, 7, False),
             "884884": (-8, 9, False), "888884": (7, 5, False)}
    for row in front.itertuples():
        dx, dy, line = place.get(row.bits, (5, 4, False))
        ax.annotate(row.bits, (row.weight_bit_cost, row.test_acc), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.8, color=INK, ha="left" if dx > 0 else "right", zorder=4,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "shrinkB": 3} if line else None)
    ax.set_xlabel("average weight bit-width  (Σ Nᵢbᵢ / Σ Nᵢ)")
    ax.set_ylabel("top-1 accuracy on test set (%)")
    ax.set_xlim(3.6, 8.15)
    ax.set_ylim(52, 105)
    # 점이 그림 전체에 퍼져 있어서 범례는 그림 아래에 둔다
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, columnspacing=1.2, handletextpad=0.4)
    ax.set_title("All 64 mixed-precision assignments (4 or 8 bits per layer), mean of 5 seeds", loc="left", color=INK)
    save(fig, out, "fig3_pareto")


def fig4_domain(summary: Path, out: Path) -> None:
    cond = pd.read_csv(summary / "domain_conditions.csv")
    table = cond[cond.task == AVG].pivot_table(index="domain", columns="condition", values="acc")
    cols = ["FP32", "W8A8", "W6A6", "W5A5", "W4A8", "W32A4", "W4A4"]
    table = table.loc[list(DOMAIN_NAMES), cols]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    im = ax.imshow(table.to_numpy(), cmap=SEQ, vmin=40, vmax=100, aspect="auto")
    ax.grid(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color="white" if v > 72 else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(table)), [DOMAIN_NAMES[d].replace("\n", " ") for d in table.index])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    # 칸 사이 흰 틈
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("top-1 accuracy (%)")
    bar.outline.set_visible(False)
    ax.set_xlabel("test input × precision (mean of 5 seeds, no re-calibration)", labelpad=8)
    save(fig, out, "fig4_domain")


def fig5_gain3d(summary: Path, out: Path) -> None:
    layers = pd.read_csv(summary / "domain_layers.csv")
    table = layers[layers.task == AVG].pivot_table(index="domain", columns="layer", values="gain_pp")
    table = table.loc[list(DOMAIN_NAMES)]
    z = table.to_numpy().clip(min=0)  # 음수 이득(최대 -0.7%p)은 막대로 그릴 수 없어 0으로 둔다
    n_dom, n_layer = z.shape
    xs, ys = np.meshgrid(np.arange(n_layer), np.arange(n_dom))
    fig = plt.figure(figsize=(5.2, 3.9))
    ax = fig.add_subplot(projection="3d")
    colors = SEQ(z.ravel() / z.max())
    ax.bar3d(xs.ravel() - 0.3, ys.ravel() - 0.3, np.zeros(z.size), 0.6, 0.6, z.ravel(), color=colors,
             edgecolor="white", linewidth=0.3, shade=True)
    ax.set_xticks(range(n_layer), [str(i) for i in range(1, 7)])
    ax.set_yticks(range(n_dom), [DOMAIN_NAMES[d].replace("\n", " ") for d in table.index], ha="left", va="center")
    ax.tick_params(axis="y", pad=-2)
    ax.set_xlabel("conv layer", labelpad=-4)
    ax.set_zlabel("restoration gain (%p)", labelpad=-2)
    ax.tick_params(axis="both", pad=-1)
    ax.view_init(elev=24, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
    ax.set_title("W4A4: accuracy regained by restoring one layer to FP32\n(negative gains drawn as 0)",
                 color=INK, y=0.97)
    fig.savefig(out / "fig5_gain3d.pdf", bbox_inches=None)
    fig.savefig(out / "fig5_gain3d.png", bbox_inches=None)
    plt.close(fig)
    print(out / "fig5_gain3d.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, default=Path("results/6conv_fold"))
    args = parser.parse_args()
    summary, out = args.exp / "summary", args.exp / "figures"
    out.mkdir(exist_ok=True)
    fig1_bits(summary, out)
    fig2_layers(summary, out)
    fig3_pareto(summary, out)
    if (summary / "domain_conditions.csv").exists():
        fig4_domain(summary, out)
        fig5_gain3d(summary, out)


if __name__ == "__main__":
    main()
