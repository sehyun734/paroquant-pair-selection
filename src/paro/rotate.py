import random
from typing import Literal, Optional

import torch
from torch import Tensor
from jaxtyping import Bool, Float32, Int64

type Arm = Literal[
    "scaling",  # rot 없이 alpha만. pair를 고르지 않으므로 build_pairs를 타지 않음
    "rand",
    "butterfly",
    "mag_diff",
    "mag_sim",
    "corr",
    "a_mag",
    "oracle",  # 대리지표 대신 목적함수를 직접 봄. 나머지 점수 arm의 상한선
]


def _oracle_score(
    w_grp: Float32[Tensor, "d_out grp_size"],
    c0: Int64[Tensor, "n"],
    c1: Int64[Tensor, "n"],
    n_angles: int = 16,
    chunk: int = 512,
) -> Float32[Tensor, "n"]:
    # 그 pair를 최적 theta로 한 번 돌렸을 때 줄어드는 rng. 다른 arm과 달리 실제로 돌려 보고 잼
    # theta는 나중에 optimize가 다시 학습하므로 여기서는 줄 세울 순위만 나오면 됨
    # ponytail: theta를 16개 그리드로 훑음. 순위가 촘촘히 갈리면 n_angles를 올릴 것

    # rot이 128채널 중 둘만 건드리므로, 나머지의 최대·최소는 상위·하위 3개만 있으면 남음
    top_v, top_i = w_grp.topk(3, dim=1)
    bot_v, bot_i = w_grp.topk(3, dim=1, largest=False)
    rng_orig = (top_v[:, 0] - bot_v[:, 0]).pow(2)

    angles = torch.arange(n_angles, device=w_grp.device) * (2 * torch.pi / n_angles)
    cos, sin = angles.cos(), angles.sin()

    # pair를 한 번에 다 잡으면 d_out × n_pairs × n_angles이라 나눠서 돎
    scores = []

    for i_chunk in range(0, len(c0), chunk):
        p0, p1 = c0[i_chunk : i_chunk + chunk], c1[i_chunk : i_chunk + chunk]

        # pair가 빼간 자리는 건너뛰고 남는 최대·최소
        keep_hi = (top_i[:, None, :] != p0[None, :, None]) & (top_i[:, None, :] != p1[None, :, None])
        keep_lo = (bot_i[:, None, :] != p0[None, :, None]) & (bot_i[:, None, :] != p1[None, :, None])
        rest_max = torch.where(keep_hi, top_v[:, None, :], -torch.inf).amax(dim=2)
        rest_min = torch.where(keep_lo, bot_v[:, None, :], torch.inf).amin(dim=2)

        w_c0, w_c1 = w_grp[:, p0], w_grp[:, p1]
        best = None

        for i_angle in range(n_angles):
            rot_c0 = cos[i_angle] * w_c0 - sin[i_angle] * w_c1
            rot_c1 = sin[i_angle] * w_c0 + cos[i_angle] * w_c1

            hi = torch.maximum(rest_max, torch.maximum(rot_c0, rot_c1))
            lo = torch.minimum(rest_min, torch.minimum(rot_c0, rot_c1))
            drop = (rng_orig[:, None] - (hi - lo).pow(2)).mean(0)  # rng이 줄수록 큼

            best = drop if best is None else torch.maximum(best, drop)

        scores.append(best)

    return torch.cat(scores)


def pair_score(
    w_grp: Float32[Tensor, "d_out grp_size"],
    a_mag_grp: Float32[Tensor, "grp_size"],
    arm: Arm,
    c0: Int64[Tensor, "n"],
    c1: Int64[Tensor, "n"],
) -> Float32[Tensor, "n"]:
    # arm이 pair를 줄 세우는 점수. 클수록 먼저 뽑힘
    # c0·c1은 그룹 로컬 인덱스
    match (arm):
        case "mag_diff":
            w_mag = w_grp.abs().mean(0)
            return (w_mag[c0] - w_mag[c1]).abs()
        case "mag_sim":
            w_mag = w_grp.abs().mean(0)
            return -(w_mag[c0] - w_mag[c1]).abs()
        case "corr":
            return torch.corrcoef(w_grp.T).nan_to_num()[c0, c1].abs()
        case "a_mag":
            w_mag = w_grp.abs().mean(0) * a_mag_grp
            return (w_mag[c0] - w_mag[c1]).abs()
        case "oracle":
            return _oracle_score(w_grp, c0, c1)
        case _:
            raise NotImplementedError(f"점수가 없는 arm: {arm}")


def _order_cands(
    w_grp: Float32[Tensor, "d_out grp_size"],
    a_mag_grp: Float32[Tensor, "grp_size"],
    arm: Arm,
    grp_size: int,
    rand_gen: random.Random,
) -> list[tuple[int, int]]:
    # 그룹 안 모든 pair를 arm이 뽑고 싶은 순서로 줄 세운 후보 목록
    c0, c1 = torch.triu_indices(grp_size, grp_size, offset=1, device=w_grp.device)

    if arm == "rand":  # 점수가 없으니 줄 세우는 대신 섞음
        cands = list(zip(c0.tolist(), c1.tolist()))
        rand_gen.shuffle(cands)
        return cands

    order = pair_score(w_grp, a_mag_grp, arm, c0, c1).argsort(descending=True)
    return list(zip(c0[order].tolist(), c1[order].tolist()))


def build_pairs(
    w: Float32[Tensor, "d_out d_in"],
    a_mag: Float32[Tensor, "d_in"],
    grp_size: int,
    n_rots: int,
    arm: Arm,
    seed: int,
    used_by_grp: Optional[dict[int, set[tuple[int, int]]]] = None,  # 그룹별 이미 쓴 pair. 밖에서 넘기면 호출을 건너 유지
    i_rot_off: int = 0,  # butterfly의 stride가 몇 번째 rot인지 알아야 함
) -> tuple[
    Int64[Tensor, "n_rots n_pairs 2"],
    Bool[Tensor, "n_rots n_pairs"],
]:
    d_in = w.shape[1]
    rand_gen = random.Random(seed)
    n_grps = d_in // grp_size
    n_pairs = d_in // 2
    if used_by_grp is None:
        used_by_grp = {}

    # 어느 arm이든 그룹의 모든 채널이 매 rot 정확히 한 pair에 들어가므로 empty가 전부 덮임
    pairs = torch.empty(n_rots, n_pairs, 2, dtype=torch.long)
    mask = torch.zeros(n_rots, n_pairs, dtype=torch.bool)

    if arm == "butterfly" and grp_size & (grp_size - 1):
        raise ValueError(f"butterfly 그룹이 2의 거듭제곱 아님: {grp_size}")
    n_strides = grp_size.bit_length() - 1

    for i_grp in range(n_grps):
        chan_off = i_grp * grp_size
        pair_off = chan_off // 2

        # 채널 번호를 stride로 XOR해 짝지음. 통계를 안 보므로 그룹마다 같은 pair
        # stride는 grp_size=128이면 7개뿐이라 8번째 rot은 첫 번째와 같은 pair를 다시 씀
        if arm == "butterfly":
            for i_rot in range(n_rots):
                stride = 1 << ((i_rot_off + i_rot) % n_strides)
                picks = [(chan, chan ^ stride) for chan in range(grp_size) if chan < chan ^ stride]  # 각 pair를 한 번만
                pair_end = pair_off + len(picks)
                pairs[i_rot, pair_off:pair_end] = chan_off + torch.tensor(picks)
            continue

        cands = _order_cands(
            w[:, chan_off : chan_off + grp_size],
            a_mag[chan_off : chan_off + grp_size],
            arm,
            grp_size,
            rand_gen,
        )

        used_pairs = used_by_grp.setdefault(i_grp, set())  # rot을 건너 유지. 같은 pair를 두 번 쓰지 않음

        for i_rot in range(n_rots):
            used_chans = set()  # rot마다 리셋. 한 rot 안에서 rot이 독립이도록
            picks = []

            # 앞에서부터 제약에 걸리지 않는 pair를 그리디로 채움
            for c0, c1 in cands:
                if len(picks) * 2 == grp_size:
                    break
                if (c0, c1) in used_pairs or c0 in used_chans or c1 in used_chans:
                    continue
                picks.append((c0, c1))
                used_chans.update((c0, c1))
                used_pairs.add((c0, c1))

            # 제약 탓에 짝을 못 찾고 남은 채널. 자리만 채우고 rot은 시키지 않음
            leftovers = [chan for chan in range(grp_size) if chan not in used_chans]
            n_picks = len(picks)
            picks += [(leftovers[i_leftover], leftovers[i_leftover + 1]) for i_leftover in range(0, len(leftovers), 2)]

            pair_end = pair_off + len(picks)
            pairs[i_rot, pair_off:pair_end] = chan_off + torch.tensor(picks)
            mask[i_rot, pair_off + n_picks : pair_end] = True  # 뒤에 붙인 leftover가 더미

    return pairs, mask


def apply_rot(
    w: Float32[Tensor, "d_out d_in"],
    pairs: Int64[Tensor, "n_rots n_pairs 2"],
    theta: Float32[Tensor, "n_rots n_pairs"],
    inv: bool = False,
) -> Float32[Tensor, "d_out d_in"]:
    # pair마다 도는 Givens rot. inv는 theta 부호와 rot 순서를 함께 뒤집음
    n_rots = len(pairs)
    order = range(n_rots - 1, -1, -1) if inv else range(n_rots)

    for i_rot in order:
        c0 = pairs[i_rot, :, 0]
        c1 = pairs[i_rot, :, 1]
        cos = torch.cos(theta[i_rot])
        sin = torch.sin(theta[i_rot])

        if inv:
            sin = -sin
        w_c0 = w[:, c0]
        w_c1 = w[:, c1]

        # c0을 제자리에서 갱신하면 c1 계산에 섞이므로 사본에 씀
        w_next = w.clone()
        w_next[:, c0] = cos * w_c0 - sin * w_c1
        w_next[:, c1] = sin * w_c0 + cos * w_c1
        w = w_next

    return w
