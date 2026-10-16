from dataclasses import dataclass, field
from itertools import groupby, product
import time
from typing import get_args

from simple_parsing import parse
import torch
from torch import Tensor

from primitives.dataset import load_pile
from primitives.givens import rotate
from primitives.hub import get_pending_jobs, make_pair_score_path, save_rows
from primitives.model import capture_linears, load_model
from primitives.pairing import Arm, build_pairs, compute_pair_score
from primitives.quantization import compute_range
from primitives.train import optimize
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    layers: list[int] = field(default_factory=lambda: [10])
    linear_names: list[str] = field(default_factory=lambda: ["self_attn.q_proj"])
    # scaling은 pair를 고르지 않으므로 볼 것이 없음.
    arms: list[str] = field(default_factory=lambda: [arm for arm in get_args(Arm) if arm != "scaling"])
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 16
    sequence_len: int = 1024
    num_rotations: list[int] = field(default_factory=lambda: [8])
    # 같은 궤적을 근사 없이 보려고 rotation.py와 같은 최적화를 여기서도 돌림.
    num_steps: list[int] = field(default_factory=lambda: [200])
    learning_rate: float = 0.01
    seeds: list[int] = field(default_factory=lambda: [42, 128, 1582, 5096, 31337])


def score_rotation(
    weight: Tensor,
    input_magnitude: Tensor,
    pairs: Tensor,
    is_dummy: Tensor,
    arm: Arm,
    group_size: int,
) -> Tensor:
    # 한 rotation이 고른 pair의 arm 점수를 group별로 냄.
    # group마다 크기 규모가 달라 하나로 뭉개지 않고, 비율은 호출부가 group별로 냄.
    # pair는 group 안에서만 지어지므로 first의 group이 곧 pair의 group임.
    buckets = {}
    for (first, second), dummy in zip(pairs.tolist(), is_dummy.tolist()):
        if not dummy:
            buckets.setdefault(first // group_size, []).append((first, second))

    scores = []
    for group_index in sorted(buckets):
        channel_offset = group_index * group_size
        picks = buckets[group_index]
        scores.append(
            compute_pair_score(
                weight[:, channel_offset : channel_offset + group_size],
                input_magnitude[channel_offset : channel_offset + group_size],
                arm,
                torch.tensor([first - channel_offset for first, _ in picks], device=weight.device),
                torch.tensor([second - channel_offset for _, second in picks], device=weight.device),
            ).mean()
        )

    return torch.stack(scores) if scores else weight.new_zeros(0)


def main() -> None:
    config = parse(Config)
    jobs = {}
    # seed는 random에만 닿으므로 다른 arm은 첫 seed로 한 번만 돌림.
    for layer, linear_name, arm in product(config.layers, config.linear_names, config.arms):
        for num_rotations, num_steps, seed in product(config.num_rotations, config.num_steps, config.seeds if arm == "random" else config.seeds[:1]):
            path = make_pair_score_path(config.model_name, layer, linear_name, config.num_bits, config.group_size, config.num_samples, config.sequence_len, num_rotations, num_steps, config.learning_rate)
            jobs[path, arm, seed] = (layer, linear_name, arm, num_rotations, num_steps, seed)
    pending = get_pending_jobs("pair_score", list(jobs))
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
            layer, linear_name, arm, num_rotations, num_steps, seed = jobs[job]
            log_time = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            x, weight = captured[layer, linear_name]
            input_magnitude = x.abs().mean(0)
            range_rtn = compute_range(weight, config.group_size)
            pairs, is_dummy = build_pairs(weight, input_magnitude, config.group_size, num_rotations, arm, seed)
            pairs, is_dummy = pairs.to(weight.device), is_dummy.to(weight.device)
            fit = optimize(weight, x, pairs, is_dummy, config.group_size, config.num_bits, num_steps, config.learning_rate)
            theta = fit.theta * ~is_dummy

            # alpha가 먼저 걸리고 rotation이 그 위에 걸리므로, rotation 전인 이 시점이 score_retention의 기준선임.
            base = weight * fit.alpha
            current = base
            rotations = []
            for rotation in range(num_rotations):
                # original_score는 pair를 고를 때 본 원본 weight로 잰 점수임.
                # score_retention은 같은 pair를 실제 rotation 시점과 rotation 전에 잰 점수의 비로, 앞선 rotation이 점수를 얼마나 남겼는지 봄.
                row = {"num_dummies": int(is_dummy[rotation].sum()), "original_score": None, "score_retention": None}
                if arm not in ("random", "butterfly"):
                    row["original_score"] = score_rotation(weight, input_magnitude, pairs[rotation], is_dummy[rotation], arm, config.group_size).mean().item()
                    score_base = score_rotation(base, input_magnitude, pairs[rotation], is_dummy[rotation], arm, config.group_size)
                    score_current = score_rotation(current, input_magnitude, pairs[rotation], is_dummy[rotation], arm, config.group_size)
                    # correlation은 기준값이 0일 수 있어 그런 group은 뺌.
                    nonzero = score_base != 0
                    if nonzero.any():
                        row["score_retention"] = (score_current[nonzero] / score_base[nonzero]).mean().item()

                current = rotate(current, pairs[rotation : rotation + 1], theta[rotation : rotation + 1])
                row["range"] = compute_range(current, config.group_size) / range_rtn
                rotations.append(row)

            result = {"initial_range": compute_range(base, config.group_size) / range_rtn, "rotations": rotations}
            # random과 butterfly는 점수가 없어 score_retention도 없음.
            score_retention = "-" if rotations[-1]["score_retention"] is None else f"{rotations[-1]['score_retention']:.1%}"
            print(f"{path} arm={arm} seed={seed} range {rotations[-1]['range']:.1%} score_retention {score_retention} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
            rows.append({"arm": arm, "seed": seed, **result})
        save_rows(path, rows)


if __name__ == "__main__":
    main()
