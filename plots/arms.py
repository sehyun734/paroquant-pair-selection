from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from primitives.plot import ARM_COLORS, compute_means, load_results, plot_grouped_means, save_figure, select, style_axis

ARMS = ["random", "butterfly", "magnitude_difference", "magnitude_similarity", "correlation", "activation_magnitude_difference"]
LAYERS = [0, 10, 20]


def main() -> None:
    rows = select(load_results("rotation"), num_steps=200, alpha_mode="joint", schedule="all", num_rotations=8)
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(18, 4.5), layout="constrained")
    for ax, layer in zip(axes, LAYERS):
        means = compute_means(select(rows, layer=layer, linear_name="self_attn.q_proj"), "loss", ("model_name",))
        plot_grouped_means(ax, {"loss": means}, ARMS)
        ax.set_title(f"layer {layer}", fontsize=10)
        style_axis(ax, "arm loss / rtn loss (%)")
        # layer마다 값의 높이가 달라 축을 따로 두고, arm 간 차이가 보이도록 가장 낮은 막대 조금 아래에서 자름.
        ax.set_ylim(bottom=min(means.values()) * 100 - 4)
    handles = [Patch(color=ARM_COLORS[arm], label=arm) for arm in ARMS]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title="q_proj num_rotations=8, 4 models mean")
    save_figure(fig, Path("figures/arms.png"))


if __name__ == "__main__":
    main()
