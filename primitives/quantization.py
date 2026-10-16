import torch
from torch import Tensor
from torch.autograd import Function


class RoundSTE(Function):
    # round는 gradient가 거의 어디서나 0이라 gradient를 그대로 흘림.
    # scale과 zero에도 gradient가 가야 alpha와 theta가 range를 줄이도록 학습됨.
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
    ) -> Tensor:
        return torch.round(x)

    @staticmethod
    def backward(
        ctx,
        grad: Tensor,
    ) -> Tensor:
        return grad


def find_grid(
    weight: Tensor,
    num_bits: int,
) -> tuple[Tensor, Tensor]:
    # range가 0인 group은 0으로 나누게 되므로 막음.
    weight_min = weight.amin(dim=-1)
    weight_max = weight.amax(dim=-1)
    scale = (weight_max - weight_min).clamp(min=1e-5) / (2**num_bits - 1)
    zero = -RoundSTE.apply(weight_min / scale)
    return scale, zero


def round_to_grid(
    weight: Tensor,
    scale: Tensor,
    zero: Tensor,
    num_bits: int,
) -> Tensor:
    level = torch.clamp(RoundSTE.apply(weight / scale) + zero, 0, 2**num_bits - 1)
    return (level - zero) * scale


def quantize(
    weight: Tensor,
    group_size: int,
    num_bits: int,
) -> Tensor:
    num_rows, num_columns = weight.shape
    if num_columns % group_size:
        raise ValueError(f"not a multiple of group_size: {num_columns} % {group_size}")
    groups = weight.reshape(num_rows, num_columns // group_size, group_size)
    scale, zero = find_grid(groups, num_bits)
    return round_to_grid(groups, scale[..., None], zero[..., None], num_bits).reshape(num_rows, num_columns)


def compute_range(
    weight: Tensor,
    group_size: int,
) -> float:
    # 양자화 오차가 range의 제곱을 따라가므로 group별 range의 제곱 평균으로 봄.
    groups = weight.reshape(weight.shape[0], -1, group_size)
    return (groups.amax(dim=2) - groups.amin(dim=2)).pow(2).mean().item()
