from pathlib import Path

import matplotlib.pyplot as plt

from primitives.plot import compute_means, load_results, plot_grouped_means, save_figure, select, style_axis

MODEL_NAMES = ["Llama-3.1-8B", "Mistral-7B-v0.3"]
# magnitude_similarity은 dummy가 많아 rotation을 덜 한 탓과 구분되지 않으므로 뺌.
ARMS = ["random", "butterfly", "magnitude_difference", "correlation", "activation_magnitude_difference"]
NUM_STEPS = [200, 800]


def main() -> None:
    rows = select(load_results("rotation"), layer=10, alpha_mode="joint", schedule="all", num_rotations=8)
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(12, 4.5), layout="constrained")
    for ax, model_name in zip(axes, MODEL_NAMES):
        means_by_group = {f"num_steps={num_steps}": compute_means(select(rows, model_name=model_name, num_steps=num_steps), "loss") for num_steps in NUM_STEPS}
        handles = plot_grouped_means(ax, means_by_group, ARMS)
        ax.set_title(model_name, fontsize=10)
        style_axis(ax, "arm loss / rtn loss (%)")
        # 모델마다 값의 높이가 달라 축을 따로 두고, arm 간 차이가 보이도록 가장 낮은 막대 조금 아래에서 자름.
        ax.set_ylim(bottom=min(value for means in means_by_group.values() for value in means.values()) * 100 - 4)
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title="q_proj layer 10 num_rotations=8")
    save_figure(fig, Path("figures/steps.png"))


if __name__ == "__main__":
    main()
