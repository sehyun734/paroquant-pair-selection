from dataclasses import dataclass
from itertools import product
from typing import Literal, Optional

import torch
from torch import Tensor
from jaxtyping import Bool, Float32, Int64

from paro.quant import fake_quant
from paro.rotate import Arm, apply_rot, build_pairs

type Alpha = Literal[
    "joint",  # theta와 함께 학습 (현행 구현)
    "off",  # 1로 묶어 rot만 남김
    "pre",  # rot 없이 미리 구한 값으로 묶음. pair를 안 보므로 arm 무관
]

# pair를 언제 고르나 × theta를 언제 학습하나, 2×2
type Sched = Literal[
    "all",  # 원본 w로 한 번  + 한꺼번에 (현행 구현)
    "seq",  # 원본 w로 한 번  + rot 하나씩. 순차화 비용만 재는 통제군
    "seq_rescore",  # rot마다 다시   + rot 하나씩
    "rescore",  # rot마다 다시   + 한꺼번에. 재채점 이득만 챙기고 순차화 비용은 안 냄
]


@dataclass
class Fit:
    loss: float  # loss가 가장 낮았던 스텝의 값
    w_pre: Float32[Tensor, "d_out d_in"]  # 그 시점의 양자화 직전 가중치
    alpha: Float32[Tensor, "d_in"]
    theta: Optional[Float32[Tensor, "n_rots n_pairs"]]  # rot이 없으면 None
    pairs: Optional[Int64[Tensor, "n_rots n_pairs 2"]]  # 쓴 pair. seq는 스스로 지으므로 여기로 돌려줌
    mask: Optional[Bool[Tensor, "n_rots n_pairs"]]


def optimize(
    w: Float32[Tensor, "d_out d_in"],
    x: Float32[Tensor, "n_toks d_in"],
    pairs: Optional[Int64[Tensor, "n_rots n_pairs 2"]],
    mask: Optional[Bool[Tensor, "n_rots n_pairs"]],
    grp_size: int,
    n_bits: int,
    n_steps: int,
    lr: float,
    alpha_fix: Optional[Float32[Tensor, "d_in"]] = None,
    theta_fix: Optional[Float32[Tensor, "n_fixed n_pairs"]] = None,  # 앞쪽 rot을 얼림. 나머지만 학습
) -> Fit:
    # Appendix A.2의 quantization-induced output error를 AdamW로 줄임
    # pairs가 없으면 rot 없이 alpha만 (scaling arm)
    use_rot = pairs is not None
    use_alpha = alpha_fix is None  # 묶으면 theta만 학습

    if use_alpha:
        alpha = torch.ones(w.shape[1], device=w.device, requires_grad=True)
    else:
        alpha = alpha_fix

    if use_rot:
        n_fixed = 0 if theta_fix is None else len(theta_fix)
        theta_new = torch.zeros(len(pairs) - n_fixed, pairs.shape[1], device=w.device, requires_grad=True)

    params = ([alpha] if use_alpha else []) + ([theta_new] if use_rot else [])
    if not params:
        raise ValueError(f"학습할 것이 없음: alpha_fix={alpha_fix is not None}, pairs=None")

    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    best = None

    for _ in range(n_steps):
        if use_rot:
            theta = theta_new if theta_fix is None else torch.cat([theta_fix, theta_new])

        # alpha → rot → 양자화 → 역rot → 역alpha. 더미는 theta를 0으로 눌러 rot 제외
        w_pre = w * alpha
        if use_rot:
            w_pre = apply_rot(w_pre, pairs, theta * ~mask)

        w_hat = fake_quant(w_pre, grp_size, n_bits)

        if use_rot:
            w_hat = apply_rot(w_hat, pairs, theta * ~mask, inv=True)
        w_hat = w_hat / alpha

        # 변환은 되돌렸으므로 남는 것은 양자화 오차가 출력에 낸 차이뿐
        loss = (x @ (w_hat - w).T).pow(2).mean()

        # 갱신 전에 담음. step 뒤에 담으면 alpha·theta만 w_pre보다 한 스텝 앞섬
        if best is None or loss.item() < best.loss:
            best = Fit(
                loss.item(),
                w_pre.detach().clone(),
                alpha.detach().clone(),  # 고정한 경우 그대로 되돌아옴
                theta.detach().clone() if use_rot else None,
                pairs,
                mask,
            )

        optim.zero_grad()
        loss.backward()
        optim.step()

    return best


def optimize_seq(
    w: Float32[Tensor, "d_out d_in"],
    x: Float32[Tensor, "n_toks d_in"],
    a_mag: Float32[Tensor, "d_in"],
    grp_size: int,
    n_bits: int,
    n_rots: int,
    n_steps: int,
    lr: float,
    arm: Arm,
    seed: int,
    alpha_fix: Float32[Tensor, "d_in"],
    rescore: bool,  # pair를 그 시점의 w로 다시 고를지
) -> Fit:
    # rot을 하나씩 확정. 매 rot "이게 마지막 rot"이라 치고 loss를 최소화하는 그리디
    # alpha가 중간에 움직이면 앞서 확정한 theta가 무효라 원래는 고정값만 받음
    # joint를 순차에 물렸을 때를 보려고 잠시 품. 단계마다 alpha를 1에서 새로 학습하므로 앞선 theta는 옛 alpha 기준으로 남음
    # if alpha_fix is None:
    #     raise ValueError("순차 sched는 alpha 고정이 필요")

    used_by_grp: dict[int, set[tuple[int, int]]] = {}  # pair 재사용 금지를 rot을 건너 유지
    pairs_hist, mask_hist = [], []
    fit = None

    for i_rot in range(n_rots):
        # 재채점은 앞선 rot까지 반영된 가중치를, 아니면 all과 같이 원본을 봄
        # 첫 rot은 되돌릴 것이 없어 둘 다 원본. 재채점의 효과는 두 번째 rot부터 갈림
        w_src = fit.w_pre if rescore and fit is not None else w
        pairs_rot, mask_rot = build_pairs(w_src, a_mag, grp_size, 1, arm, seed, used_by_grp, i_rot)

        pairs_hist.append(pairs_rot.to(w.device))
        mask_hist.append(mask_rot.to(w.device))

        # 앞선 rot의 theta는 얼리고 이번 것만 학습
        theta_fix = None if fit is None else fit.theta
        fit = optimize(w, x, torch.cat(pairs_hist), torch.cat(mask_hist), grp_size, n_bits, n_steps, lr, alpha_fix, theta_fix)

    return fit


def alpha_fixes(
    alphas: list[str],
    w: Float32[Tensor, "d_out d_in"],
    x: Float32[Tensor, "n_toks d_in"],
    grp_size: int,
    n_bits: int,
    n_steps: list[int],
    lr: float,
) -> dict[
    tuple[Alpha, int],  # (alpha, n_steps)
    Optional[Float32[Tensor, "d_in"]],  # optimize에 넘길 alpha_fix
]:
    # alpha 모드마다 alpha를 무엇으로 묶을지. pair를 안 보므로 arm 무관하고 미리 한 번만 구하면 됨
    # pre만 최적화가 필요하고 n_steps마다 값이 달라짐
    fixes = {}

    for alpha, steps in product(alphas, n_steps):
        if alpha == "joint":
            fixes[alpha, steps] = None  # 묶지 않고 theta와 함께 학습
        elif alpha == "off":
            fixes[alpha, steps] = torch.ones(w.shape[1], device=w.device)
        else:
            fixes[alpha, steps] = optimize(w, x, None, None, grp_size, n_bits, steps, lr).alpha

    return fixes
