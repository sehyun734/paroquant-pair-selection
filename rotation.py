from dataclasses import dataclass, field
from itertools import groupby, product
import time
from typing import Literal, get_args

from simple_parsing import parse
import torch
from torch import Tensor

from primitives.dataset import load_pile
from primitives.hub import get_pending_jobs, make_rotation_path, save_rows
from primitives.model import capture_linears, load_model
from primitives.pairing import Arm, build_pairs
from primitives.quantization import compute_range, quantize
from primitives.train import Fit, optimize
from primitives.utils import format_duration, format_memory

# joint는 theta와 함께 학습하는 현행 구현, off는 1로 묶어 rotation만 남김, pretrained는 rotation 없이 미리 학습한 값으로 묶음.
AlphaMode = Literal["joint", "off", "pretrained"]
# pair를 언제 고르는지 × theta를 언제 학습하는지의 2×2.
# all은 원본 weight로 한 번 고르고 한꺼번에 학습하는 현행 구현, sequential은 원본으로 한 번 고르고 rotation 하나씩 학습함.
# sequential_rescore는 rotation마다 다시 고르고 하나씩 학습하고, rescore는 그렇게 다시 고른 pair로 한꺼번에 학습함.
Schedule = Literal["all", "sequential", "sequential_rescore", "rescore"]


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    layers: list[int] = field(default_factory=lambda: [0, 10, 20])
    linear_names: list[str] = field(default_factory=lambda: ["self_attn.q_proj"])
    arms: list[str] = field(default_factory=lambda: list(get_args(Arm)))
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 16
    sequence_len: int = 1024
    alpha_modes: list[str] = field(default_factory=lambda: ["joint"])
    schedules: list[str] = field(default_factory=lambda: ["all"])
    num_rotations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_steps: list[int] = field(default_factory=lambda: [200])
    learning_rate: float = 0.01
    seeds: list[int] = field(default_factory=lambda: [42, 128, 1582, 5096, 31337])


def optimize_sequential(
    weight: Tensor,
    x: Tensor,
    input_magnitude: Tensor,
    group_size: int,
    num_bits: int,
    num_rotations: int,
    num_steps: int,
    learning_rate: float,
    arm: Arm,
    seed: int,
    alpha_fixed: Tensor | None,
    rescores: bool,
) -> Fit:
    # rotation을 하나씩 확정하며, 매번 이게 마지막 rotation이라 치고 loss를 줄이는 greedy임.
    # alpha가 joint면 단계마다 alpha를 1에서 새로 학습하므로, 앞선 theta는 옛 alpha 기준으로 남음.
    used_pairs_by_group = {}
    pairs_history, is_dummy_history = [], []
    fit = None

    for rotation in range(num_rotations):
        # rescores면 앞선 rotation까지 반영된 weight로, 아니면 all과 같이 원본으로 고름.
        # 첫 rotation은 되돌릴 것이 없어 둘 다 원본이므로, 재채점의 효과는 두 번째부터 갈림.
        source = fit.transformed if rescores and fit is not None else weight
        pairs, is_dummy = build_pairs(source, input_magnitude, group_size, 1, arm, seed, used_pairs_by_group, rotation)
        pairs_history.append(pairs.to(weight.device))
        is_dummy_history.append(is_dummy.to(weight.device))

        # 앞선 rotation의 theta는 얼리고 이번 것만 학습함.
        theta_fixed = None if fit is None else fit.theta
        fit = optimize(weight, x, torch.cat(pairs_history), torch.cat(is_dummy_history), group_size, num_bits, num_steps, learning_rate, alpha_fixed, theta_fixed)

    return fit


def make_fixed_alphas(
    alpha_modes: list[AlphaMode],
    weight: Tensor,
    x: Tensor,
    group_size: int,
    num_bits: int,
    num_steps: list[int],
    learning_rate: float,
) -> dict[tuple[AlphaMode, int], Tensor | None]:
    # alpha를 무엇으로 묶을지는 pair를 안 보므로 arm과 무관하게 한 번만 구함.
    # pretrained만 학습이 필요하고 num_steps마다 값이 달라짐.
    fixed_alphas = {}
    for alpha_mode in alpha_modes:
        for steps in num_steps:
            if alpha_mode == "joint":
                fixed_alphas[alpha_mode, steps] = None
            elif alpha_mode == "off":
                fixed_alphas[alpha_mode, steps] = torch.ones(weight.shape[1], device=weight.device)
            else:
                fixed_alphas[alpha_mode, steps] = optimize(weight, x, None, None, group_size, num_bits, steps, learning_rate).alpha
    return fixed_alphas


def make_jobs(
    config: Config,
) -> dict[tuple[str, str, int], tuple]:
    # 닿지 않는 축은 None으로 두어 같은 조건을 여러 번 돌리지 않음.
    # scaling은 rotation이 없으므로 alpha_mode, schedule, num_rotations가 없음.
    # seed는 random에만 닿으므로 다른 arm은 첫 seed로 한 번만 돌림.
    jobs = {}
    for layer, linear_name, arm in product(config.layers, config.linear_names, config.arms):
        uses_rotation = arm != "scaling"
        for alpha_mode, schedule, num_rotations, num_steps, seed in product(
            config.alpha_modes if uses_rotation else [None],
            config.schedules if uses_rotation else [None],
            config.num_rotations if uses_rotation else [None],
            config.num_steps,
            config.seeds if arm == "random" else config.seeds[:1],
        ):
            path = make_rotation_path(config.model_name, layer, linear_name, config.num_bits, config.group_size, config.num_samples, config.sequence_len, alpha_mode, schedule, num_rotations, num_steps, config.learning_rate)
            jobs[path, arm, seed] = (layer, linear_name, arm, alpha_mode, schedule, num_rotations, num_steps, seed)
    return jobs


def main() -> None:
    config = parse(Config)
    jobs = make_jobs(config)
    pending = get_pending_jobs("rotation", list(jobs))
    if not pending:
        return

    model, tokenizer = load_model(config.model_name)
    samples = load_pile(tokenizer, config.num_samples, config.sequence_len)
    captured = capture_linears(model, samples, config.layers, config.linear_names)
    del model

    for (layer, linear_name), (x, weight) in captured.items():
        target_jobs = [job for job in pending if jobs[job][:2] == (layer, linear_name)]
        if not target_jobs:
            continue

        # 변환 없이 양자화한 rtn을 모든 값의 100% 기준으로 씀.
        loss_rtn = (x @ (quantize(weight, config.group_size, config.num_bits) - weight).T).pow(2).mean().item()
        range_rtn = compute_range(weight, config.group_size)
        input_magnitude = x.abs().mean(0)
        fixed_alphas = make_fixed_alphas(config.alpha_modes, weight, x, config.group_size, config.num_bits, config.num_steps, config.learning_rate)

        # 설정 하나의 arm과 seed를 모두 돌린 뒤 파일 하나로 올림.
        for path, path_jobs in groupby(sorted(target_jobs), key=lambda job: job[0]):
            rows = []
            for job in path_jobs:
                _, _, arm, alpha_mode, schedule, num_rotations, num_steps, seed = jobs[job]
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

                if arm == "scaling":
                    fit = optimize(weight, x, None, None, config.group_size, config.num_bits, num_steps, config.learning_rate)
                    total_steps = num_steps
                elif schedule == "all":
                    pairs, is_dummy = build_pairs(weight, input_magnitude, config.group_size, num_rotations, arm, seed)
                    fit = optimize(weight, x, pairs.to(weight.device), is_dummy.to(weight.device), config.group_size, config.num_bits, num_steps, config.learning_rate, fixed_alphas[alpha_mode, num_steps])
                    total_steps = num_steps
                else:
                    # sequential 계열은 rotation마다 num_steps씩 학습하므로 총 학습량이 num_rotations배임.
                    fit = optimize_sequential(weight, x, input_magnitude, config.group_size, config.num_bits, num_rotations, num_steps, config.learning_rate, arm, seed, fixed_alphas[alpha_mode, num_steps], schedule != "sequential")
                    total_steps = num_steps * num_rotations
                    # rescore는 pair만 가져오고 theta는 all과 같이 한꺼번에 다시 학습함.
                    if schedule == "rescore":
                        fit = optimize(weight, x, fit.pairs, fit.is_dummy, config.group_size, config.num_bits, num_steps, config.learning_rate, fixed_alphas[alpha_mode, num_steps])
                        total_steps += num_steps

                # range는 양자화기가 실제로 본 weight로 잼.
                weight_range = compute_range(fit.transformed, config.group_size)
                duration = time.perf_counter() - log_time
                result = {
                    "loss": fit.loss / loss_rtn,
                    "range": weight_range / range_rtn,
                    "num_dummies": None if fit.is_dummy is None else int(fit.is_dummy.sum()),
                    "total_steps": total_steps,
                    "duration": duration,
                }
                print(f"{path} arm={arm} seed={seed} loss {result['loss']:.1%} range {result['range']:.1%} time {format_duration(duration)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                rows.append({"arm": arm, "seed": seed, **result})
            save_rows(path, rows)


if __name__ == "__main__":
    main()
