import torch
from torch import Tensor
from jaxtyping import Float32


class STE(torch.autograd.Function):
    # round는 기울기가 거의 어디서나 0이라 그래디언트를 그대로 흘림
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def fake_quant(
    w: Float32[Tensor, "d_out d_in"],
    grp_size: int,
    n_bits: int,
) -> Float32[Tensor, "d_out d_in"]:
    # 그룹별 asymmetric min-max 양자화
    d_out, d_in = w.shape
    if d_in % grp_size:
        raise ValueError(f"그룹 배수 아님: {d_in} % {grp_size}")

    n_levels = 2**n_bits - 1
    n_grps = d_in // grp_size
    w_grps = w.reshape(d_out, n_grps, grp_size)

    # scale이 rng에 비례. rng이 0인 그룹은 0으로 나누므로 막음
    grp_min = w_grps.amin(dim=2, keepdim=True)
    grp_max = w_grps.amax(dim=2, keepdim=True)
    scale = (grp_max - grp_min).clamp(min=1e-5) / n_levels
    zp = -STE.apply(grp_min / scale)

    w_int = (STE.apply(w_grps / scale) + zp).clamp(0, n_levels)
    return ((w_int - zp) * scale).reshape(d_out, d_in)


def grp_stats(
    w: Float32[Tensor, "d_out d_in"],
    grp_size: int,
) -> tuple[
    float,  # rng
    float,  # kurt
]:
    d_out = w.shape[0]
    w_grps = w.reshape(d_out, -1, grp_size)

    # 오차가 rng의 제곱을 따라가므로 제곱 평균
    rng = w_grps.amax(dim=2) - w_grps.amin(dim=2)

    # excess kurtosis. 정규분포가 0, 클수록 outlier가 튀어나온 그룹
    w_ctr = w_grps - w_grps.mean(dim=2, keepdim=True)
    var = w_ctr.pow(2).mean(dim=2)
    kurt = w_ctr.pow(4).mean(dim=2) / var.pow(2).clamp(min=1e-12) - 3

    return rng.pow(2).mean().item(), kurt.mean().item()
