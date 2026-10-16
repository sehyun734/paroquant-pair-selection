from pathlib import Path
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from primitives.plot import ARM_COLORS, annotate_bar, load_results, save_figure, select, style_axis

MODEL_NAMES = ["Qwen3-0.6B", "Llama-3.1-8B", "Mistral-7B-v0.3", "gemma-2-9b"]
LAYERS = [10, 20]
# sequential_rescore는 빗금으로 all과 구분함.
SERIES = [("all", "random"), ("sequential_rescore", "random"), ("sequential_rescore", "oracle")]


def main() -> None:
    # alpha_mode를 현행 구현인 joint로 풀고, 현행 all random과 나란히 둠.
    rows = select(load_results("rotation"), alpha_mode="joint", num_rotations=8, num_steps=200)
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(14, 4.5), layout="constrained")
    width = 0.8 / len(SERIES)
    for ax, layer in zip(axes, LAYERS):
        losses = []
        for index, (schedule, arm) in enumerate(SERIES):
            values = [mean(row["loss"] for row in select(rows, model_name=model_name, layer=layer, schedule=schedule, arm=arm)) * 100 for model_name in MODEL_NAMES]
            values = [*values, mean(values)]
            losses += values
            positions = [x - 0.4 + width * (index + 0.5) for x in range(len(values))]
            ax.bar(positions, values, width=width, color=ARM_COLORS[arm], hatch="//" if schedule == "sequential_rescore" else None, edgecolor="white", zorder=3)
            for position, value in zip(positions, values):
                annotate_bar(ax, position, value, f"{value:.1f}", fontsize=7)
        ax.set_xticks(range(len(MODEL_NAMES) + 1), [*MODEL_NAMES, "mean"], fontsize=8)
        ax.set_title(f"layer {layer}", fontsize=10)
        style_axis(ax, "arm loss / rtn loss (%)")
        # 막대 간 차이가 보이도록 가장 낮은 막대 조금 아래에서 자름.
        ax.set_ylim(bottom=min(losses) - 4)
    handles = [Patch(facecolor=ARM_COLORS[arm], hatch="//" if schedule == "sequential_rescore" else None, edgecolor="white", label=f"{schedule} {arm}") for schedule, arm in SERIES]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title="q_proj num_rotations=8 alpha_mode=joint")
    save_figure(fig, Path("figures/joint_sequential.png"))

if __name__ == "__main__":
    main()
