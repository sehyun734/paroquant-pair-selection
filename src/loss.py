from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import get_args

from simple_parsing import parse

from paro.calib import capture_all
from paro.optimize import Alpha, Sched, alpha_fixes, optimize, optimize_seq
from paro.quant import fake_quant, grp_stats
from paro.rotate import Arm, build_pairs


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
    layers: list[int] = field(default_factory=lambda: [0, 10, 20])
    linears: list[str] = field(default_factory=lambda: ["self_attn.q_proj"])
    arms: list[str] = field(default_factory=lambda: list(get_args(Arm.__value__)))
    alphas: list[str] = field(default_factory=lambda: ["joint"])
    scheds: list[str] = field(default_factory=lambda: ["all"])
    n_seqs: int = 16
    seq_len: int = 1024
    n_bits: int = 4
    n_rots: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    grp_size: int = 128
    seeds: list[int] = field(default_factory=lambda: [42, 128, 1582, 5096, 31337])
    n_steps: list[int] = field(default_factory=lambda: [200])
    lr: float = 0.01

    def __post_init__(self):
        unknown = set(self.arms) - set(get_args(Arm.__value__))
        if unknown:
            raise ValueError(f"모르는 arm: {sorted(unknown)}")
        unknown = set(self.alphas) - set(get_args(Alpha.__value__))
        if unknown:
            raise ValueError(f"모르는 alpha: {sorted(unknown)}")
        unknown = set(self.scheds) - set(get_args(Sched.__value__))
        if unknown:
            raise ValueError(f"모르는 sched: {sorted(unknown)}")
        # 순차는 alpha가 중간에 움직이면 앞서 확정한 theta가 무효
        # if set(self.scheds) - {"all"} and not set(self.alphas) - {"joint"}:
        #     raise ValueError("순차 sched는 고정 alpha가 필요: --alphas off 또는 pre")


def main():
    cfg = parse(Config)

    print(
        f"{'model':<18}{'layer':<6}{'linear':<18}{'arm':<10}"
        f"{'alpha':<9}{'sched':<13}{'seed':<7}{'n_rots':<7}{'n_steps':<8}"
        f"{'loss':>8}"  # 출력 loss, RTN 대비
        f"{'dummy':>8}"  # 자리만 채워 rot하지 않는 pair 수
        f"{'rng':>8}"  # 그룹 rng, RTN 대비
        f"{'kurt':>8}",  # 그룹 excess kurtosis
        flush=True,
    )

    for model, (layer, linear), (x, w) in capture_all(cfg.models, cfg.layers, cfg.linears, cfg.n_seqs, cfg.seq_len):
        # 변환 없이 양자화한 rtn. 이후 모든 값의 100% 기준
        w_rtn = fake_quant(w, cfg.grp_size, cfg.n_bits)
        loss_rtn = (x @ (w_rtn - w).T).pow(2).mean().item()
        rng_rtn, _ = grp_stats(w, cfg.grp_size)
        a_mag = x.abs().mean(0)

        # alpha 모드마다 무엇으로 묶을지. pair를 안 보므로 arm 무관하고 한 번만 구하면 됨
        fixes = alpha_fixes(cfg.alphas, w, x, cfg.grp_size, cfg.n_bits, cfg.n_steps, cfg.lr)

        for arm in cfg.arms:
            use_rot = arm != "scaling"  # scaling은 rot 없이 alpha만
            # 안 닿는 축은 첫 값으로 고정. 같은 조건을 여러 번 돌리지 않음
            for sched, alpha, seed, n_rots, n_steps in product(
                cfg.scheds if use_rot else ["all"],  # rot이 없으면 순차도 없음
                cfg.alphas if use_rot else ["joint"],  # scaling은 묶으면 학습할 것이 안 남음
                cfg.seeds if arm == "rand" else cfg.seeds[:1],
                cfg.n_rots if use_rot else cfg.n_rots[:1],
                cfg.n_steps,
            ):
                # if sched != "all" and alpha == "joint":  # 순차는 alpha 고정이 필요
                #     continue

                alpha_fix = fixes[alpha, n_steps] if use_rot else None

                if not use_rot:
                    fit = optimize(w, x, None, None, cfg.grp_size, cfg.n_bits, n_steps, cfg.lr)
                elif sched == "all":
                    pairs, mask = build_pairs(w, a_mag, cfg.grp_size, n_rots, arm, seed)
                    fit = optimize(w, x, pairs.to(w.device), mask.to(w.device), cfg.grp_size, cfg.n_bits, n_steps, cfg.lr, alpha_fix)
                else:
                    fit = optimize_seq(w, x, a_mag, cfg.grp_size, cfg.n_bits, n_rots, n_steps, cfg.lr, arm, seed, alpha_fix, rescore=sched != "seq")
                    if sched == "rescore":
                        # pair만 가져오고 theta는 all과 같이 한꺼번에 다시 학습
                        fit = optimize(w, x, fit.pairs, fit.mask, cfg.grp_size, cfg.n_bits, n_steps, cfg.lr, alpha_fix)

                # 양자화기가 실제로 본 가중치의 통계
                rng, kurt = grp_stats(fit.w_pre, cfg.grp_size)

                # 안 닿은 축에 값을 찍으면 돌린 것처럼 읽힘
                label = f"{Path(model).name:<18}{layer:<6}{linear:<18}{arm:<10}"
                shown_alpha = alpha if use_rot else "-"
                shown_sched = sched if use_rot else "-"
                shown_dummy = str(int(fit.mask.sum())) if use_rot else "-"
                shown_seed = str(seed) if arm == "rand" else "-"
                shown_rots = str(n_rots) if use_rot else "-"
                print(
                    f"{label:<52}{shown_alpha:<9}{shown_sched:<13}{shown_seed:<7}{shown_rots:<7}{n_steps:<8}" f"{fit.loss / loss_rtn:>8.1%}{shown_dummy:>8}" f"{rng / rng_rtn:>8.1%}{kurt:>8.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
