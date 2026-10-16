import random
from typing import Literal

import torch
from torch import Tensor

# scaling은 rotation 없이 alpha만 써서 pair를 고르지 않음.
# oracle은 대리지표 대신 목적함수를 직접 봄.
Arm = Literal["scaling", "random", "butterfly", "magnitude_difference", "magnitude_similarity", "correlation", "activation_magnitude_difference", "oracle"]


def compute_oracle_score(
    group: Tensor,
    first: Tensor,
    second: Tensor,
    num_angles: int = 16,
    chunk_size: int = 512,
) -> Tensor:
    # pair를 가장 좋은 각도로 한 번 돌렸을 때 줄어드는 range를 점수로 봄.
    # theta는 나중에 다시 학습하므로 여기서는 순위만 나오면 됨.
    # 따라서 각도는 num_angles개 grid로만 훑음.
    angles = torch.arange(num_angles, device=group.device) * (2 * torch.pi / num_angles)
    cos, sin = angles.cos(), angles.sin()

    # rotation은 group 안 두 channel만 바꾸므로, 나머지의 최대·최소는 상위·하위 3개만 있으면 남음.
    top_values, top_indices = group.topk(3, dim=1)
    bottom_values, bottom_indices = group.topk(3, dim=1, largest=False)
    original_range = (top_values[:, 0] - bottom_values[:, 0]).pow(2)

    # pair를 한 번에 다 잡으면 rows × pairs × angles라 나눠서 돎.
    scores = []
    for start in range(0, len(first), chunk_size):
        chunk_first = first[start : start + chunk_size]
        chunk_second = second[start : start + chunk_size]

        # pair가 빼간 자리는 건너뛰고 남는 최대·최소.
        keeps_top = (top_indices[:, None, :] != chunk_first[None, :, None]) & (top_indices[:, None, :] != chunk_second[None, :, None])
        keeps_bottom = (bottom_indices[:, None, :] != chunk_first[None, :, None]) & (bottom_indices[:, None, :] != chunk_second[None, :, None])
        rest_max = torch.where(keeps_top, top_values[:, None, :], -torch.inf).amax(dim=2)
        rest_min = torch.where(keeps_bottom, bottom_values[:, None, :], torch.inf).amin(dim=2)

        first_columns = group[:, chunk_first]
        second_columns = group[:, chunk_second]
        best = None
        for angle_cos, angle_sin in zip(cos, sin):
            rotated_first = angle_cos * first_columns - angle_sin * second_columns
            rotated_second = angle_sin * first_columns + angle_cos * second_columns
            high = torch.maximum(rest_max, torch.maximum(rotated_first, rotated_second))
            low = torch.minimum(rest_min, torch.minimum(rotated_first, rotated_second))
            drop = (original_range[:, None] - (high - low).pow(2)).mean(0)
            best = drop if best is None else torch.maximum(best, drop)
        scores.append(best)

    return torch.cat(scores)


def compute_pair_score(
    group: Tensor,
    group_input_magnitude: Tensor,
    arm: Arm,
    first: Tensor,
    second: Tensor,
) -> Tensor:
    # arm이 pair를 줄 세우는 점수로, 클수록 먼저 뽑힘.
    # first, second는 group 안 channel index임.
    if arm == "magnitude_difference":
        magnitude = group.abs().mean(0)
        return (magnitude[first] - magnitude[second]).abs()
    if arm == "magnitude_similarity":
        magnitude = group.abs().mean(0)
        return -(magnitude[first] - magnitude[second]).abs()
    if arm == "correlation":
        return torch.corrcoef(group.T).nan_to_num()[first, second].abs()
    if arm == "activation_magnitude_difference":
        magnitude = group.abs().mean(0) * group_input_magnitude
        return (magnitude[first] - magnitude[second]).abs()
    if arm == "oracle":
        return compute_oracle_score(group, first, second)
    raise NotImplementedError(f"arm without score: {arm}")


def order_candidates(
    group: Tensor,
    group_input_magnitude: Tensor,
    arm: Arm,
    group_size: int,
    generator: random.Random,
) -> list[tuple[int, int]]:
    # group 안 모든 pair를 arm이 뽑고 싶은 순서로 줄 세움.
    first, second = torch.triu_indices(group_size, group_size, offset=1, device=group.device)

    # random은 점수가 없으므로 줄 세우는 대신 섞음.
    if arm == "random":
        candidates = list(zip(first.tolist(), second.tolist()))
        generator.shuffle(candidates)
        return candidates

    order = compute_pair_score(group, group_input_magnitude, arm, first, second).argsort(descending=True)
    return list(zip(first[order].tolist(), second[order].tolist()))


def build_pairs(
    weight: Tensor,
    input_magnitude: Tensor,
    group_size: int,
    num_rotations: int,
    arm: Arm,
    seed: int,
    used_pairs_by_group: dict[int, set[tuple[int, int]]] | None = None,
    rotation_offset: int = 0,
) -> tuple[Tensor, Tensor]:
    # used_pairs_by_group를 밖에서 넘기면 호출을 건너 이미 쓴 pair를 유지함.
    # rotation_offset은 butterfly가 몇 번째 rotation의 stride를 쓸지 정함.
    num_columns = weight.shape[1]
    generator = random.Random(seed)
    if used_pairs_by_group is None:
        used_pairs_by_group = {}
    if arm == "butterfly" and group_size & (group_size - 1):
        raise ValueError(f"group_size is not a power of 2: {group_size}")
    num_strides = group_size.bit_length() - 1

    # 어느 arm이든 group의 모든 channel이 매 rotation 정확히 한 pair에 들어가므로 empty가 전부 덮임.
    pairs = torch.empty(num_rotations, num_columns // 2, 2, dtype=torch.long)
    is_dummy = torch.zeros(num_rotations, num_columns // 2, dtype=torch.bool)

    for group_index in range(num_columns // group_size):
        channel_offset = group_index * group_size
        pair_offset = channel_offset // 2

        # butterfly는 channel 번호를 stride로 XOR해 짝지어, 통계를 안 보고 group마다 같은 pair를 씀.
        # stride는 group_size=128이면 7개뿐이라 8번째 rotation은 첫 번째와 같은 pair를 다시 씀.
        if arm == "butterfly":
            for rotation in range(num_rotations):
                stride = 1 << ((rotation_offset + rotation) % num_strides)
                picks = [(channel, channel ^ stride) for channel in range(group_size) if channel < channel ^ stride]
                pairs[rotation, pair_offset : pair_offset + len(picks)] = channel_offset + torch.tensor(picks)
            continue

        candidates = order_candidates(
            weight[:, channel_offset : channel_offset + group_size],
            input_magnitude[channel_offset : channel_offset + group_size],
            arm,
            group_size,
            generator,
        )
        # 같은 pair를 두 번 쓰지 않도록 rotation을 건너 유지함.
        used_pairs = used_pairs_by_group.setdefault(group_index, set())

        for rotation in range(num_rotations):
            # rotation끼리 독립이도록 한 rotation 안에서는 channel을 한 번씩만 씀.
            used_channels = set()
            picks = []
            for first, second in candidates:
                if len(picks) * 2 == group_size:
                    break
                if (first, second) in used_pairs or first in used_channels or second in used_channels:
                    continue
                picks.append((first, second))
                used_channels.update((first, second))
                used_pairs.add((first, second))

            # 제약 탓에 짝을 못 찾은 channel은 자리만 채우고 rotation하지 않는 dummy로 둠.
            leftovers = [channel for channel in range(group_size) if channel not in used_channels]
            num_picks = len(picks)
            picks += [(leftovers[index], leftovers[index + 1]) for index in range(0, len(leftovers), 2)]
            pairs[rotation, pair_offset : pair_offset + len(picks)] = channel_offset + torch.tensor(picks)
            is_dummy[rotation, pair_offset + num_picks : pair_offset + len(picks)] = True

    return pairs, is_dummy
