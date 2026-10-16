from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from primitives.plot import ARM_COLORS, load_results, save_figure, select, style_axis, style_rotation_axis

ARMS = ["random", "butterfly", "magnitude_difference", "correlation", "activation_magnitude_difference"]
NUM_ROTATIONS = [1, 2, 4, 8]


def main() -> None:
    rows = select(load_results("connectivity"), layer=[10, 20])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ["blocks", "lambda2"], ["blocks", "lambda2"]):
        for arm in ARMS:
            values = [mean(row[metric] for row in select(rows, arm=arm, num_rotations=num_rotations)) for num_rotations in NUM_ROTATIONS]
            # magnitude_difference와 activation_magnitude_difference는 값이 같아 겹치므로 activation_magnitude_difference를 점선으로 그림.
            ax.plot(NUM_ROTATIONS, values, marker="o", color=ARM_COLORS[arm], linestyle="--" if arm == "activation_magnitude_difference" else "-", label=arm)
        ax.set_title(title, fontsize=10)
        style_rotation_axis(ax, NUM_ROTATIONS)
        style_axis(ax, metric)
    # blocks는 64에서 1까지 줄어 선형 축이면 1~4 사이 차이가 안 보이므로 log 축에 정수 눈금을 씀.
    axes[0].set_yscale("log", base=2)
    axes[0].set_yticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
    handles = axes[0].get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title="q_proj layer 10, 20, 4 models mean")
    save_figure(fig, Path("figures/connectivity.png"))


if __name__ == "__main__":
    main()
