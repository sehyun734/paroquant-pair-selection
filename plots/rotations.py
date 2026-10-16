from pathlib import Path

import matplotlib.pyplot as plt

from primitives.plot import load_results, plot_rotation_means, save_figure, select, style_axis

# magnitude_similarity은 dummy가 많아 rotation을 덜 한 탓과 구분되지 않으므로 뺌.
ARMS = ["random", "butterfly", "magnitude_difference", "correlation", "activation_magnitude_difference", "oracle"]
NUM_ROTATIONS = [1, 2, 4, 8]
LAYERS = [10, 20]


def main() -> None:
    rows = select(load_results("rotation"), layer=LAYERS, alpha_mode="joint", schedule="all", num_steps=200)
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(11, 4.5), layout="constrained")
    for ax, layer in zip(axes, LAYERS):
        plot_rotation_means(ax, select(rows, layer=layer), "loss", ARMS, NUM_ROTATIONS)
        ax.set_title(f"layer {layer}", fontsize=10)
        style_axis(ax, "arm loss / rtn loss (%)")
    handles = axes[0].get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title="q_proj, 4 models mean")
    save_figure(fig, Path("figures/rotations.png"))


if __name__ == "__main__":
    main()
