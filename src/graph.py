from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

import torch
from simple_parsing import parse

from paro.calib import capture_all
from paro.rotate import Arm, build_pairs


@dataclass
class Config:
    models: list[str] = field(
        default_factory=lambda: [
            "meta-llama/Llama-3.1-8B",
            "Qwen/Qwen3-0.6B",
            "mistralai/Mistral-7B-v0.3",
            "google/gemma-2-9b",
        ]
    )
    layers: list[int] = field(default_factory=lambda: [0, 10, 20])
    linears: list[str] = field(default_factory=lambda: ["self_attn.q_proj"])
    arms: list[str] = field(default_factory=lambda: list(get_args(Arm.__value__)))
    n_seqs: int = 16
    seq_len: int = 1024
    grp_size: int = 128
    seed: int = 42
    n_rots: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])

    def __post_init__(self):
        unknown = set(self.arms) - set(get_args(Arm.__value__))
        if unknown:
            raise ValueError(f"모르는 arm: {sorted(unknown)}")


def main():
    cfg = parse(Config)

    print(
        f"{'model':<18}{'layer':<6}{'linear':<18}{'arm':<10}{'n_rots':<7}"
        f"{'blocks':>8}"  # 채널이 몇 덩어리로 갈라지는지
        f"{'dist':>7}"  # 같은 덩어리 안에서 평균 몇 다리 건너 닿는지
        f"{'lam2':>7}",  # 덩어리를 둘로 쪼개기가 얼마나 어려운지
        flush=True,
    )

    for model, (layer, linear), (x, w) in capture_all(cfg.models, cfg.layers, cfg.linears, cfg.n_seqs, cfg.seq_len):
        d_in = x.shape[1]
        a_mag = x.abs().mean(0)

        for arm in cfg.arms:
            if arm == "scaling":  # pair를 고르지 않으니 그래프도 없음
                continue
            for n_rots in cfg.n_rots:
                pairs, mask = build_pairs(w, a_mag, cfg.grp_size, n_rots, arm, cfg.seed)

                # rot하지 않는 더미는 엣지가 아님
                live = [
                    (c0, c1)
                    for i_rot in range(n_rots)
                    for (c0, c1), is_dummy in zip(pairs[i_rot].tolist(), mask[i_rot].tolist())
                    if not is_dummy
                ]

                n_blocks = 0
                dists_all = []
                lam2s = []
                # pair는 그룹 안에서만 지어지므로 그래프도 그룹별로 따로
                for chan_off in range(0, d_in, cfg.grp_size):
                    # 전 rot의 pair를 겹쳐 엣지로
                    # butterfly는 stride가 돌아 같은 pair가 다시 나오므로 set으로 중복 제거
                    edges = {
                        (c0 - chan_off, c1 - chan_off) for c0, c1 in live if chan_off <= c0 < chan_off + cfg.grp_size
                    }

                    # nbrs는 BFS용 인접 리스트, adj는 Laplacian용 인접 행렬
                    nbrs = [[] for _ in range(cfg.grp_size)]
                    adj = torch.zeros(cfg.grp_size, cfg.grp_size, dtype=torch.float64)
                    for c0, c1 in edges:
                        nbrs[c0].append(c1)
                        nbrs[c1].append(c0)
                        adj[c0, c1] = adj[c1, c0] = 1

                    def reach(start_chan: int) -> dict[int, int]:
                        # start_chan에서 닿는 채널과 그 거리
                        dists = {start_chan: 0}
                        queue = deque([start_chan])
                        while queue:
                            chan = queue.popleft()
                            for nxt_chan in nbrs[chan]:
                                if nxt_chan not in dists:
                                    dists[nxt_chan] = dists[chan] + 1
                                    queue.append(nxt_chan)
                        return dists

                    # 아직 안 닿은 채널에서 새로 시작할 때마다 덩어리 하나
                    seen = set()
                    for start_chan in range(cfg.grp_size):
                        if start_chan not in seen:
                            n_blocks += 1
                            seen |= set(reach(start_chan))

                    # 자기 자신(0)과 못 닿는 채널은 빠지므로 덩어리 안 거리만 모임
                    for start_chan in range(cfg.grp_size):
                        dists_all += [dist for dist in reach(start_chan).values() if dist > 0]

                    lap = torch.diag(adj.sum(1)) - adj
                    lam2s.append(torch.linalg.eigvalsh(lap)[1].item())  # 오름차순, [0]은 항상 0

                # 그룹별로 쌓은 값을 그룹 수로 나눠 평균
                n_grps = d_in // cfg.grp_size
                label = f"{Path(model).name:<18}{layer:<6}{linear:<18}{arm:<10}"
                print(
                    f"{label:<52}{n_rots:<7}"
                    f"{n_blocks / n_grps:>8.1f}"
                    f"{sum(dists_all) / max(len(dists_all), 1):>7.2f}"  # 엣지가 없으면 빔
                    f"{sum(lam2s) / n_grps:>7.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
