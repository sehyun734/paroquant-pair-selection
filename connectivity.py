from dataclasses import dataclass, field
from itertools import groupby, product
from typing import get_args

from simple_parsing import parse
import torch
from torch import Tensor

from primitives.dataset import load_pile
from primitives.hub import get_pending_jobs, make_connectivity_path, save_rows
from primitives.model import capture_linears, load_model
from primitives.pairing import Arm, build_pairs


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    layers: list[int] = field(default_factory=lambda: [0, 10, 20])
    linear_names: list[str] = field(default_factory=lambda: ["self_attn.q_proj"])
    # scaling은 pair를 고르지 않으므로 그래프도 없음.
    arms: list[str] = field(default_factory=lambda: [arm for arm in get_args(Arm) if arm != "scaling"])
    group_size: int = 128
    num_samples: int = 16
    sequence_len: int = 1024
    num_rotations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    seeds: list[int] = field(default_factory=lambda: [42])


def compute_connectivity(
    pairs: Tensor,
    is_dummy: Tensor,
    num_columns: int,
    group_size: int,
) -> dict[str, float]:
    # 모든 rotation의 pair를 겹쳐 edge로 보고, rotation하지 않는 dummy는 edge에서 뺌.
    live_pairs = [tuple(pair) for pair, dummy in zip(pairs.flatten(0, 1).tolist(), is_dummy.flatten().tolist()) if not dummy]
    num_blocks = 0
    lambda2s = []

    # pair는 group 안에서만 지어지므로 그래프도 group별로 따로 봄.
    for channel_offset in range(0, num_columns, group_size):
        # butterfly는 stride가 돌아 같은 pair가 다시 나오므로 set으로 중복을 없앰.
        edges = {(first - channel_offset, second - channel_offset) for first, second in live_pairs if channel_offset <= first < channel_offset + group_size}
        adjacency = torch.zeros(group_size, group_size, dtype=torch.float64)
        for first, second in edges:
            adjacency[first, second] = adjacency[second, first] = 1

        # Laplacian의 고유값 0의 개수가 덩어리 수이고, 두 번째로 작은 고유값이 lambda2임.
        # 고유값은 오름차순이고 첫 번째는 항상 0임.
        laplacian = torch.diag(adjacency.sum(1)) - adjacency
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        num_blocks += int((eigenvalues < 1e-8).sum())
        lambda2s.append(eigenvalues[1].item())

    # blocks는 channel이 몇 덩어리로 갈라지는지, lambda2는 덩어리를 둘로 쪼개기가 얼마나 어려운지.
    num_groups = num_columns // group_size
    return {"blocks": num_blocks / num_groups, "lambda2": sum(lambda2s) / num_groups}


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    jobs = {}
    # seed는 random에만 닿으므로 다른 arm은 첫 seed로 한 번만 돌림.
    for layer, linear_name, arm in product(config.layers, config.linear_names, config.arms):
        for num_rotations, seed in product(config.num_rotations, config.seeds if arm == "random" else config.seeds[:1]):
            path = make_connectivity_path(config.model_name, layer, linear_name, config.group_size, config.num_samples, config.sequence_len, num_rotations)
            jobs[path, arm, seed] = (layer, linear_name, arm, num_rotations, seed)
    pending = get_pending_jobs("connectivity", list(jobs))
    if not pending:
        return

    model, tokenizer = load_model(config.model_name)
    samples = load_pile(tokenizer, config.num_samples, config.sequence_len)
    captured = capture_linears(model, samples, config.layers, config.linear_names)
    del model

    # 설정 하나의 arm과 seed를 모두 돌린 뒤 파일 하나로 올림.
    for path, path_jobs in groupby(sorted(pending), key=lambda job: job[0]):
        rows = []
        for job in path_jobs:
            layer, linear_name, arm, num_rotations, seed = jobs[job]
            x, weight = captured[layer, linear_name]
            pairs, is_dummy = build_pairs(weight, x.abs().mean(0), config.group_size, num_rotations, arm, seed)
            result = compute_connectivity(pairs, is_dummy, weight.shape[1], config.group_size)
            print(f"{path} arm={arm} seed={seed} blocks {result['blocks']:.1f} lambda2 {result['lambda2']:.2f}")
            rows.append({"arm": arm, "seed": seed, **result})
        save_rows(path, rows)


if __name__ == "__main__":
    main()
