from math import sqrt
from pathlib import Path
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from primitives.plot import annotate_bar, load_results, save_figure, select, style_axis

# rebuttal Q4 표의 저자 값으로, scaling은 alpha만, +8 IR은 그 위에 독립 rotation 8개를 붙인 것임.
AUTHOR_VALUES = {"self_attn.q_proj": (78, 53), "mlp.gate_proj": (75, 55)}


def main() -> None:
    rows = select(load_results("rotation"), model_name="Qwen3-0.6B", layer=0, num_steps=200)
    fig, axes = plt.subplots(1, len(AUTHOR_VALUES), figsize=(10, 4.5), layout="constrained")
    for ax, (linear_name, (author_scaling, author_rotation)) in zip(axes, AUTHOR_VALUES.items()):
        scaling = select(rows, linear_name=linear_name, arm="scaling")[0]["loss"] * 100
        random_losses = [row["loss"] * 100 for row in select(rows, linear_name=linear_name, arm="random", alpha_mode="joint", schedule="all", num_rotations=8)]
        # A.2 표기대로 l2로 재면 mse 비율의 제곱근이 됨.
        bars = {
            "scaling\n(author)": (author_scaling, "tab:gray"),
            "scaling\nl2": (sqrt(scaling / 100) * 100, "tab:orange"),
            "scaling\nmse": (scaling, "tab:blue"),
            "+8 IR\n(author)": (author_rotation, "tab:gray"),
            "random 8\nmse": (mean(random_losses), "tab:blue"),
        }
        for x, (label, (value, color)) in enumerate(bars.items()):
            ax.bar(x, value, color=color, width=0.6, zorder=3)
            annotate_bar(ax, x, value, f"{value:.1f}")
        ax.errorbar(len(bars) - 1, mean(random_losses), yerr=[[mean(random_losses) - min(random_losses)], [max(random_losses) - mean(random_losses)]], color="black", capsize=4, zorder=4)
        ax.set_xticks(range(len(bars)), list(bars), fontsize=9)
        ax.set_title(linear_name, fontsize=10)
        style_axis(ax, "loss / rtn loss (%)")
    handles = [Patch(color=color, label=label) for label, color in [("author", "tab:gray"), ("l2", "tab:orange"), ("mse", "tab:blue")]]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title="Qwen3-0.6B layer 0 reproduction, random error bar is seed min~max")
    save_figure(fig, Path("figures/reproduction.png"))


if __name__ == "__main__":
    main()
