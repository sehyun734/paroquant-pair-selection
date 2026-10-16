from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from huggingface_hub import snapshot_download
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from primitives.hub import get_repo_name

ARM_COLORS = {
    "scaling": "tab:gray",
    "random": "tab:blue",
    "butterfly": "tab:cyan",
    "magnitude_difference": "tab:orange",
    "magnitude_similarity": "tab:brown",
    "correlation": "tab:green",
    "activation_magnitude_difference": "tab:red",
    "oracle": "tab:purple",
}
GRID_COLOR = "#e5e4df"


def parse_value(
    value: str,
) -> int | float | str:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_results(
    entry_name: str,
) -> list[dict]:
    # 경로의 모델과 설정을 풀어, 파일 안의 arm과 seed별 행마다 붙임.
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=[f"{entry_name}/*/*.json"])) / entry_name
    rows = []
    for file in sorted(result_dir.glob("*/*.json")):
        settings = {"model_name": file.parent.name}
        for setting in file.stem.split(","):
            name, value = setting.split("=", 1)
            settings[name] = parse_value(value)
        rows.extend(settings | row for row in json.loads(file.read_text()))
    return rows


def select(
    rows: list[dict],
    **conditions: object,
) -> list[dict]:
    # 경로에 없는 설정은 None으로 봄.
    return [row for row in rows if all(row.get(name) in (value if isinstance(value, list) else [value]) for name, value in conditions.items())]


def compute_means(
    rows: list[dict],
    metric: str,
    cells: tuple[str, ...] = ("model_name", "layer"),
) -> dict[str, float]:
    # random은 seed마다 값이 달라 cell마다 seed 평균을 먼저 낸 뒤, arm마다 cell 평균을 냄.
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        values[row["arm"]][tuple(row[name] for name in cells)].append(row[metric])
    return {arm: mean(mean(cell_values) for cell_values in arm_values.values()) for arm, arm_values in values.items()}


def annotate_bar(
    ax: Axes,
    x: float,
    value: float,
    text: str,
    fontsize: int = 9,
) -> None:
    # 음수 막대는 아래로 자라므로 글자도 막대 끝 아래에 둠.
    ax.annotate(text, (x, value), xytext=(0, 6 if value >= 0 else -6), textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", fontsize=fontsize)


def plot_grouped_means(
    ax: Axes,
    means_by_group: dict[str, dict[str, float]],
    arms: list[str],
) -> list[Patch]:
    # arm마다 조건별 막대를 나란히 두고, 조건은 진하기로 구분해 그 진하기의 handle을 돌려줌.
    width = 0.6 / len(means_by_group)
    handles = []
    for index, (group, means) in enumerate(means_by_group.items()):
        alpha = 1.0 if len(means_by_group) == 1 else 0.4 + 0.6 * index / (len(means_by_group) - 1)
        handles.append(Patch(color="tab:gray", alpha=alpha, label=group))
        for x, arm in enumerate(arms):
            position = x - 0.3 + width * (index + 0.5)
            ax.bar(position, means[arm] * 100, width=width, color=ARM_COLORS[arm], alpha=alpha, zorder=3)
            annotate_bar(ax, position, means[arm] * 100, f"{means[arm] * 100:.1f}", fontsize=9 if len(means_by_group) == 1 else 7)
    # 이름이 길어 겹치므로 _에서 줄을 바꿈.
    ax.set_xticks(range(len(arms)), [arm.replace("_", "\n") for arm in arms], fontsize=9)
    return handles


def plot_rotation_means(
    ax: Axes,
    rows: list[dict],
    metric: str,
    arms: list[str],
    num_rotations: list[int],
) -> None:
    # rotation 수마다 arm의 평균을 선으로 이음.
    # random은 기준선이라 다른 선에 가리지 않도록 굵게 맨 위에 그림.
    results = [compute_means(select(rows, num_rotations=rotations), metric) for rotations in num_rotations]
    for arm in arms:
        is_random = arm == "random"
        ax.plot(num_rotations, [means[arm] * 100 for means in results], marker="o", color=ARM_COLORS[arm], linewidth=3 if is_random else 1.5, zorder=5 if is_random else 3, label=arm)
    style_rotation_axis(ax, num_rotations)


def style_rotation_axis(
    ax: Axes,
    num_rotations: list[int],
) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(num_rotations, num_rotations)
    ax.set_xlabel("num_rotations")


def style_axis(
    ax: Axes,
    ylabel: str,
) -> None:
    ax.set_ylabel(ylabel)
    ax.margins(y=0.15)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(
    fig: Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"saved {path}")
