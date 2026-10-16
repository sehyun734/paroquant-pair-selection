from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import AdamW

from primitives.givens import rotate
from primitives.quantization import quantize


@dataclass
class Fit:
    loss: float
    # loss가 가장 낮았던 step의 양자화 직전 weight.
    transformed: Tensor
    alpha: Tensor
    theta: Tensor | None
    pairs: Tensor | None
    is_dummy: Tensor | None


def optimize(
    weight: Tensor,
    x: Tensor,
    pairs: Tensor | None,
    is_dummy: Tensor | None,
    group_size: int,
    num_bits: int,
    num_steps: int,
    learning_rate: float,
    alpha_fixed: Tensor | None = None,
    theta_fixed: Tensor | None = None,
) -> Fit:
    # Appendix A.2의 quantization-induced output error를 AdamW로 줄임.
    # pairs가 없으면 rotation 없이 alpha만 학습하고, theta_fixed를 넘기면 앞쪽 rotation은 얼림.
    uses_rotation = pairs is not None
    learns_alpha = alpha_fixed is None
    alpha = torch.ones(weight.shape[1], device=weight.device, requires_grad=True) if learns_alpha else alpha_fixed
    if uses_rotation:
        num_fixed = 0 if theta_fixed is None else len(theta_fixed)
        theta_new = torch.zeros(len(pairs) - num_fixed, pairs.shape[1], device=weight.device, requires_grad=True)

    parameters = ([alpha] if learns_alpha else []) + ([theta_new] if uses_rotation else [])
    if not parameters:
        raise ValueError("nothing to learn")
    optimizer = AdamW(parameters, lr=learning_rate, weight_decay=0)

    best = None
    for _ in range(num_steps):
        if uses_rotation:
            theta = theta_new if theta_fixed is None else torch.cat([theta_fixed, theta_new])

        # alpha, rotation, 양자화, 역rotation, 역alpha 순으로 감.
        # dummy는 theta를 0으로 눌러 rotation에서 뺌.
        transformed = weight * alpha
        if uses_rotation:
            transformed = rotate(transformed, pairs, theta * ~is_dummy)
        dequantized = quantize(transformed, group_size, num_bits)
        if uses_rotation:
            dequantized = rotate(dequantized, pairs, theta * ~is_dummy, inverse=True)
        dequantized = dequantized / alpha

        # 변환은 되돌렸으므로 남는 것은 양자화 오차가 출력에 낸 차이뿐임.
        loss = (x @ (dequantized - weight).T).pow(2).mean()

        # step 뒤에 담으면 alpha와 theta만 transformed보다 한 step 앞서므로 갱신 전에 담음.
        if best is None or loss.item() < best.loss:
            best = Fit(loss.item(), transformed.detach().clone(), alpha.detach().clone(), theta.detach().clone() if uses_rotation else None, pairs, is_dummy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return best
