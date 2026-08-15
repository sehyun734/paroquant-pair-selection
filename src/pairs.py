from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import get_args

import torch
from torch import Tensor
from jaxtyping import Bool, Float32, Int64
from simple_parsing import parse

from paro.calib import capture_all
from paro.optimize import Alpha, alpha_fixes, optimize
from paro.quant import grp_stats
from paro.rotate import Arm, apply_rot, build_pairs, pair_score


@dataclass
class Config:
    models: list[str] = field(
        default_factory=lambda: [
            "Qwen/Qwen3-0.6B",
            "meta-llama/Llama-3.1-8B",
            "mistralai/Mistral-7B-v0.3",
            "google/gemma-2-9b",
        ]
    )
    layers: list[int] = field(default_factory=lambda: [10])
    linears: list[str] = field(default_factory=lambda: ["self_attn.q_proj"])
    arms: list[str] = field(default_factory=lambda: list(get_args(Arm.__value__)))
    alphas: list[str] = field(default_factory=lambda: ["joint"])
    n_seqs: int = 16
    seq_len: int = 1024
    grp_size: int = 128
    seeds: list[int] = field(default_factory=lambda: [42, 128, 1582, 5096, 31337])
    # 같은 궤적을 근사 없이 보려고 loss.py와 같은 최적화를 여기서도 돌림
    n_bits: int = 4
    n_rots: list[int] = field(default_factory=lambda: [8])
    n_steps: list[int] = field(default_factory=lambda: [200])
    lr: float = 0.01

    def __post_init__(self):
        unknown = set(self.arms) - set(get_args(Arm.__value__))
        if unknown:
            raise ValueError(f"모르는 arm: {sorted(unknown)}")
        unknown = set(self.alphas) - set(get_args(Alpha.__value__))
        if unknown:
            raise ValueError(f"모르는 alpha: {sorted(unknown)}")


def score_rot(
    w: Float32[Tensor, "d_out d_in"],
    a_mag: Float32[Tensor, "d_in"],
    pairs_rot: Int64[Tensor, "n_pairs 2"],
    mask_rot: Bool[Tensor, "n_pairs"],
    arm: Arm,
    grp_size: int,
) -> Float32[Tensor, "n_grps"]:
    # 한 rot이 고른 pair의 arm 점수. 클수록 그 arm이 먼저 뽑는 pair
    # 그룹마다 크기 규모가 달라 하나로 뭉개지 않음. 비율은 호출부가 그룹별로 냄

    # rot하지 않는 더미는 빼고 그룹별로 모음
    # pair는 그룹 안에서만 지어지므로 c0의 그룹이 곧 pair의 그룹
    buckets: dict[int, list[tuple[int, int]]] = {}
    for (c0, c1), is_dummy in zip(pairs_rot.tolist(), mask_rot.tolist()):
        if not is_dummy:
            buckets.setdefault(c0 // grp_size, []).append((c0, c1))

    # pair_score는 그룹 로컬 인덱스를 받으므로 chan_off를 빼서 넘김
    scores = []

    for i_grp in sorted(buckets):
        chan_off = i_grp * grp_size
        picks = buckets[i_grp]

        scores.append(
            pair_score(
                w[:, chan_off : chan_off + grp_size],
                a_mag[chan_off : chan_off + grp_size],
                arm,
                torch.tensor([c0 - chan_off for c0, _ in picks], device=w.device),
                torch.tensor([c1 - chan_off for _, c1 in picks], device=w.device),
            ).mean()
        )

    return torch.stack(scores) if scores else w.new_zeros(0)  # 전부 더미면 빈 텐서


def main():
    cfg = parse(Config)

    # dummy~mag_diff는 rot 직전 상태, rng~drng는 rot 직후 상태
    print(
        f"{'model':<18}{'layer':<6}{'linear':<18}{'arm':<10}"
        f"{'alpha':<9}{'seed':<7}{'n_rots':<7}{'n_steps':<8}"
        f"{'i_rot':<7}"  # 0은 rot을 아직 안 한 시점
        f"{'dummy':>7}"  # 이 rot에서 자리만 채운 pair 수
        f"{'orig':>10}"  # arm 점수. pair를 고를 때 본 w
        f"{'base':>10}"  # 같은 pair, rot 전 w*alpha
        f"{'cur':>10}"  # 같은 pair, 앞선 rot을 다 겪은 w_cur
        f"{'hit':>7}"  # cur/base. 앞선 rot이 점수를 얼마나 남겼는지. mag_sim만 방향 반대
        f"{'mag_diff':>10}"  # 크기 차이. arm 무관 공통 축
        f"{'rng':>8}"  # 누적 rng, RTN 대비
        f"{'drng':>8}",  # 이 rot 하나가 줄인 몫
        flush=True,
    )

    for model, (layer, linear), (x, w) in capture_all(cfg.models, cfg.layers, cfg.linears, cfg.n_seqs, cfg.seq_len):
        a_mag = x.abs().mean(0)
        rng_rtn, _ = grp_stats(w, cfg.grp_size)  # loss.py와 같은 100% 기준

        # alpha 모드마다 무엇으로 묶을지. pair를 안 보므로 arm 무관하고 한 번만 구하면 됨
        fixes = alpha_fixes(cfg.alphas, w, x, cfg.grp_size, cfg.n_bits, cfg.n_steps, cfg.lr)

        for arm in cfg.arms:
            if arm == "scaling":  # pair를 고르지 않으니 볼 것이 없음
                continue
            has_score = arm not in ("rand", "butterfly")  # 점수 없는 arm은 orig/base/hit 무의미

            for alpha, seed, n_rots, n_steps in product(
                cfg.alphas,
                cfg.seeds if arm == "rand" else cfg.seeds[:1],  # seed는 rand에만 닿음
                cfg.n_rots,
                cfg.n_steps,
            ):
                pairs, mask = build_pairs(w, a_mag, cfg.grp_size, n_rots, arm, seed)
                pairs, mask = pairs.to(w.device), mask.to(w.device)

                fit = optimize(w, x, pairs, mask, cfg.grp_size, cfg.n_bits, n_steps, cfg.lr, fixes[alpha, n_steps])
                theta = fit.theta * ~mask  # 더미는 rot하지 않음

                # alpha가 먼저, rot이 그 위. rot 전 이 시점이 hit·drng의 기준선
                w_base = w * fit.alpha
                w_cur = w_base
                rng_cur, _ = grp_stats(w_cur, cfg.grp_size)

                shown_seed = str(seed) if arm == "rand" else "-"  # 안 닿은 seed는 안 찍음
                label = (
                    f"{Path(model).name:<18}{layer:<6}{linear:<18}{arm:<10}"
                    f"{alpha:<9}{shown_seed:<7}{n_rots:<7}{n_steps:<8}"
                )

                # rot 전 기준선. pair를 안 썼으니 rng 말고는 잴 것이 없음
                print(
                    f"{label}{0:<7}{'-':>7}{'-':>10}{'-':>10}{'-':>10}{'-':>7}{'-':>10}"
                    f"{rng_cur / rng_rtn:>8.1%}{'-':>8}",
                    flush=True,
                )

                for i_rot in range(n_rots):
                    # 같은 pair를 세 시점의 w로 재봄. orig는 pair를 고를 때, base는 rot 전, cur는 실제 rot 시점
                    if has_score:
                        score_orig = score_rot(w, a_mag, pairs[i_rot], mask[i_rot], arm, cfg.grp_size)
                        score_base = score_rot(w_base, a_mag, pairs[i_rot], mask[i_rot], arm, cfg.grp_size)
                        score_cur = score_rot(w_cur, a_mag, pairs[i_rot], mask[i_rot], arm, cfg.grp_size)
                    mag_diff = score_rot(w_cur, a_mag, pairs[i_rot], mask[i_rot], "mag_diff", cfg.grp_size)

                    # rot을 먼저 걸어야 이 rot이 rng를 얼마나 줄였는지 나옴
                    w_cur = apply_rot(w_cur, pairs[i_rot : i_rot + 1], theta[i_rot : i_rot + 1])
                    rng_next, _ = grp_stats(w_cur, cfg.grp_size)

                    shown_orig = f"{score_orig.mean().item():.2e}" if has_score else "-"
                    shown_base = f"{score_base.mean().item():.2e}" if has_score else "-"
                    shown_cur = f"{score_cur.mean().item():.2e}" if has_score else "-"
                    shown_hit = "-"
                    if has_score:
                        base_ok = score_base != 0  # corr은 기준값이 0일 수 있음
                        if base_ok.any():
                            hit = (score_cur[base_ok] / score_base[base_ok]).mean()
                            shown_hit = f"{hit.item():.0%}"
                    print(
                        f"{label}{i_rot + 1:<7}"
                        f"{int(mask[i_rot].sum()):>7}"
                        f"{shown_orig:>10}{shown_base:>10}{shown_cur:>10}{shown_hit:>7}"
                        f"{mag_diff.mean().item():>10.2e}"
                        f"{rng_next / rng_rtn:>8.1%}{rng_next / rng_cur:>8.1%}",
                        flush=True,
                    )

                    rng_cur = rng_next

                # 재구성한 궤적이 optimize가 본 것과 같은지
                assert torch.allclose(w_cur, fit.w_pre, rtol=1e-4, atol=1e-5), "궤적 불일치"


if __name__ == "__main__":
    main()
