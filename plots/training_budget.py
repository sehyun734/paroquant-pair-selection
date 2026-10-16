from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from primitives.plot import ARM_COLORS, annotate_bar, load_results, save_figure, select, style_axis

LAYERS = [10, 20]
TOTAL_STEPS = [200, 1600]
# sequential_rescore는 빗금으로 all과 구분함.
SERIES = [("all", "random"), ("all", "oracle"), ("sequential_rescore", "random"), ("sequential_rescore", "oracle")]


def main() -> None:
    # paro_pair의 결과라 rotation 결과와 섞지 않고 그 안에서만 비교함.
    rows = select(load_results("training_budget"), alpha_mode="joint", num_rotations=8)
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(11, 4.5), layout="constrained", sharey=True)
    width = 0.8 / len(SERIES)
    losses = []
    for ax, layer in zip(axes, LAYERS):
        for index, (schedule, arm) in enumerate(SERIES):
            for x, total_steps in enumerate(TOTAL_STEPS):
                loss = select(rows, layer=layer, schedule=schedule, arm=arm, total_steps=total_steps)[0]["loss"] * 100
                losses.append(loss)
                position = x - 0.4 + width * (index + 0.5)
                ax.bar(position, loss, width=width, color=ARM_COLORS[arm], hatch="//" if schedule == "sequential_rescore" else None, edgecolor="white", zorder=3)
                annotate_bar(ax, position, loss, f"{loss:.1f}", fontsize=8)
        ax.set_xticks(range(len(TOTAL_STEPS)), [f"total_steps={total_steps}" for total_steps in TOTAL_STEPS], fontsize=9)
        ax.set_title(f"layer {layer}", fontsize=10)
        style_axis(ax, "arm loss / rtn loss (%)")
    # 막대 간 차이가 보이도록 가장 낮은 막대 조금 아래에서 자름.
    axes[0].set_ylim(bottom=min(losses) - 4)
    handles = [Patch(facecolor=ARM_COLORS[arm], hatch="//" if schedule == "sequential_rescore" else None, edgecolor="white", label=f"{schedule} {arm}") for schedule, arm in SERIES]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title="paro_pair Qwen3-0.6B q_proj num_rotations=8 alpha_mode=joint, random seed 42")
    save_figure(fig, Path("figures/training_budget.png"))


if __name__ == "__main__":
    main()
