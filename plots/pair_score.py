from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from primitives.plot import ARM_COLORS, load_results, save_figure, select, style_axis

# random과 butterfly는 점수가 없어 score_retention도 없음.
# magnitude_similarity은 dummy가 많아 rotation을 덜 한 탓과 구분되지 않으므로 뺌.
ARMS = ["magnitude_difference", "correlation", "activation_magnitude_difference", "oracle"]
NUM_ROTATIONS = 8


def main() -> None:
    rows = select(load_results("pair_score"), model_name="Llama-3.1-8B", layer=10, num_rotations=NUM_ROTATIONS, num_steps=200)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for arm in ARMS:
        arm_rows = select(rows, arm=arm)
        score_retentions = [mean(row["rotations"][rotation]["score_retention"] for row in arm_rows) * 100 for rotation in range(NUM_ROTATIONS)]
        ax.plot(range(1, NUM_ROTATIONS + 1), score_retentions, marker="o", markersize=3, color=ARM_COLORS[arm], label=arm)
    ax.set_xlabel("rotation")
    style_axis(ax, "score_retention (%)")
    handles = ax.get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, title=f"Llama-3.1-8B q_proj layer 10 num_rotations={NUM_ROTATIONS}")
    save_figure(fig, Path("figures/pair_score.png"))


if __name__ == "__main__":
    main()
