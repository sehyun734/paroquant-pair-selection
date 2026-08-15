"""PYTHONPATH=src python src/test_paro.py — 모델 없이 도는 자체 점검."""

import torch

from paro.optimize import alpha_fixes, optimize, optimize_seq
from paro.quant import fake_quant, grp_stats
from paro.rotate import _oracle_score, apply_rot, build_pairs, pair_score

GRP, D_IN, D_OUT = 8, 16, 12
ARMS = ["rand", "butterfly", "mag_diff", "mag_sim", "corr", "a_mag", "oracle"]


def toy():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, D_IN, generator=g)
    w = torch.randn(D_OUT, D_IN, generator=g)
    w[:, 3] *= 8  # outlier 채널 하나
    return x, w, x.abs().mean(0)


def test_build_pairs():
    x, w, a_mag = toy()
    for arm in ARMS:
        for n_rots in (1, 2, 4):
            pairs, mask = build_pairs(w, a_mag, GRP, n_rots, arm, seed=42)
            assert pairs.shape == (n_rots, D_IN // 2, 2), (arm, n_rots, pairs.shape)
            assert mask.shape == (n_rots, D_IN // 2)

            used_pairs = set()
            for i_rot in range(n_rots):
                chans = pairs[i_rot].flatten().tolist()
                # 한 rot 안에서 모든 채널이 정확히 한 번. rot끼리 독립이어야 함
                assert sorted(chans) == list(range(D_IN)), (arm, i_rot, sorted(chans))

                for (c0, c1), is_dummy in zip(pairs[i_rot].tolist(), mask[i_rot].tolist()):
                    assert c0 // GRP == c1 // GRP, f"{arm}: pair가 그룹을 넘음 {c0},{c1}"
                    # butterfly만 stride가 돌아 같은 pair를 다시 씀. 나머지는 후보에서 지움
                    if is_dummy or arm == "butterfly":
                        continue
                    key = (c0, c1)
                    assert key not in used_pairs, f"{arm}: 같은 pair 재사용 {key}"
                    used_pairs.add(key)

    # butterfly는 XOR로 늘 짝이 떨어지므로 더미가 없음
    _, mask = build_pairs(w, a_mag, GRP, 4, "butterfly", seed=42)
    assert not mask.any(), "butterfly는 자리를 늘 채움"

    # 크기가 비슷한 pair부터 쓰는 mag_sim은 후보가 말라 더미가 생김
    _, mask = build_pairs(w, a_mag, GRP, 4, "mag_sim", seed=42)
    assert mask.any(), "mag_sim은 더미가 나와야 함"


def test_oracle_score():
    x, w, _ = toy()
    w_grp = w[:, :GRP]
    c0, c1 = torch.triu_indices(GRP, GRP, offset=1)

    got = _oracle_score(w_grp, c0, c1, chunk=5)  # 청크 경계를 넘겨 붙이는지도 같이 봄
    assert got.shape == c0.shape

    # 상위·하위 3개만 쓰는 지름길이 맞는지, apply_rot으로 실제로 돌려 rng를 다시 재서 대조
    n_angles = 16
    rng_orig = (w_grp.amax(1) - w_grp.amin(1)).pow(2)
    for i_pair in range(len(c0)):
        pair = torch.tensor([[[c0[i_pair], c1[i_pair]]]])
        drops = []
        for i_angle in range(n_angles):
            theta = torch.tensor([[i_angle * (2 * torch.pi / n_angles)]])
            rot = apply_rot(w_grp, pair, theta)
            drops.append((rng_orig - (rot.amax(1) - rot.amin(1)).pow(2)).mean())
        want = max(drops)
        assert torch.allclose(got[i_pair], want, atol=1e-4), (i_pair, got[i_pair].item(), want.item())

    # 목적함수를 직접 보므로 크기 차이 기준보다 rng를 더 줄이는 pair를 1순위로 뽑아야 함
    a_mag = x.abs().mean(0)[:GRP]
    top_oracle = got.argmax()
    top_mag = pair_score(w_grp, a_mag, "mag_diff", c0, c1).argmax()
    assert got[top_oracle] >= got[top_mag]


def test_apply_rot_roundtrip():
    x, w, a_mag = toy()
    pairs, mask = build_pairs(w, a_mag, GRP, 4, "rand", seed=42)
    theta = torch.randn(*pairs.shape[:2]) * ~mask

    back = apply_rot(apply_rot(w, pairs, theta), pairs, theta, inv=True)
    assert torch.allclose(back, w, atol=1e-5), "inv가 원상복구를 못 함"

    # rot은 직교변환이라 행 노름이 보존됨
    rot = apply_rot(w, pairs, theta)
    assert torch.allclose(rot.norm(dim=1), w.norm(dim=1), atol=1e-5), "rot이 노름을 바꿈"

    # 더미는 theta를 눌렀으므로 그 채널은 그대로
    dummy_chans = pairs[0][mask[0]].flatten()
    if len(dummy_chans):
        assert torch.allclose(apply_rot(w, pairs[:1], theta[:1])[:, dummy_chans], w[:, dummy_chans], atol=1e-6)


def test_fake_quant():
    x, w, _ = toy()
    w_q = fake_quant(w, GRP, n_bits=4)
    w_grps = w.reshape(D_OUT, -1, GRP)
    step = (w_grps.amax(2) - w_grps.amin(2)).clamp(min=1e-5) / 15  # 4비트는 15칸
    err = (w_q - w).abs().reshape(D_OUT, -1, GRP).amax(2)
    assert (err <= step / 2 + 1e-6).all(), "오차가 반 스텝 초과"

    # 비트를 늘리면 오차와 rng가 단조로 좋아짐
    errs = [(fake_quant(w, GRP, b) - w).pow(2).mean().item() for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2], errs
    assert grp_stats(w, GRP)[0] > 0


def test_alpha_fixes():
    x, w, _ = toy()
    fixes = alpha_fixes(["joint", "off", "pre"], w, x, GRP, 4, [2, 30], lr=0.01)

    assert set(fixes) == {(a, s) for a in ("joint", "off", "pre") for s in (2, 30)}
    assert fixes["joint", 2] is None, "joint는 묶지 않음"
    assert torch.equal(fixes["off", 2], torch.ones(D_IN)), "off는 1로 묶음"
    # pre는 최적화로 구하므로 1도 아니고 n_steps마다 값이 달라짐. dict를 n_steps로 키잉하는 이유
    assert fixes["pre", 2].shape == (D_IN,)
    assert not torch.equal(fixes["pre", 2], torch.ones(D_IN))
    assert not torch.equal(fixes["pre", 2], fixes["pre", 30])

    # 요청 안 한 모드는 최적화를 돌리지 않음
    assert set(alpha_fixes(["joint"], w, x, GRP, 4, [2], lr=0.01)) == {("joint", 2)}


def test_build_pairs_resumable():
    # used_by_grp를 밖에서 넘겨 한 rot씩 지으면 한꺼번에 지은 것과 같아야 함. 순차 sched가 이것에 기댐
    x, w, a_mag = toy()
    for arm in ARMS:
        want_pairs, want_mask = build_pairs(w, a_mag, GRP, 4, arm, seed=42)

        used_by_grp = {}
        got = [build_pairs(w, a_mag, GRP, 1, arm, 42, used_by_grp, i_rot) for i_rot in range(4)]
        got_pairs = torch.cat([p for p, _ in got])
        got_mask = torch.cat([m for _, m in got])

        assert torch.equal(got_pairs, want_pairs), arm
        assert torch.equal(got_mask, want_mask), arm


def test_optimize_seq():
    x, w, a_mag = toy()
    alpha_fix = torch.ones(D_IN)

    # rot이 하나면 순차와 동시가 같은 문제. loss가 일치해야 함
    for arm in ("rand", "oracle"):
        pairs, mask = build_pairs(w, a_mag, GRP, 1, arm, seed=42)
        want = optimize(w, x, pairs, mask, GRP, 4, 20, 0.01, alpha_fix)
        got = optimize_seq(w, x, a_mag, GRP, 4, 1, 20, 0.01, arm, 42, alpha_fix, rescore=False)
        assert abs(got.loss - want.loss) < 1e-9, (arm, got.loss, want.loss)

    # 재채점을 안 하면 pair가 all과 같고, 하면 갈라짐
    pairs_all, _ = build_pairs(w, a_mag, GRP, 4, "oracle", seed=42)
    pairs_seq = optimize_seq(w, x, a_mag, GRP, 4, 4, 20, 0.01, "oracle", 42, alpha_fix, rescore=False).pairs
    pairs_re = optimize_seq(w, x, a_mag, GRP, 4, 4, 20, 0.01, "oracle", 42, alpha_fix, rescore=True).pairs
    assert torch.equal(pairs_seq, pairs_all), "재채점 안 하면 all과 같은 pair여야 함"
    assert torch.equal(pairs_re[0], pairs_all[0]), "첫 rot은 되돌릴 것이 없어 같아야 함"
    assert not torch.equal(pairs_re, pairs_all), "재채점하면 뒤쪽 rot이 갈려야 함"

    # 재채점한 pair가 그 시점의 가중치에서 실제로 더 나은지. rot 1은 양쪽이 같으므로 상태도 같음
    fit1 = optimize_seq(w, x, a_mag, GRP, 4, 1, 20, 0.01, "oracle", 42, alpha_fix, rescore=False)
    w_cur = fit1.w_pre[:, :GRP]

    def gain(pairs_rot):  # 첫 그룹이 rot 2에서 고른 pair의 rng 감소분
        sel = pairs_rot[pairs_rot[:, 0] < GRP]
        return _oracle_score(w_cur, sel[:, 0], sel[:, 1]).sum()

    assert gain(pairs_re[1]) > gain(pairs_seq[1]), "재채점이 더 나은 pair를 골라야 함"

    # 앞선 rot의 theta는 얼려야 함
    fit = optimize_seq(w, x, a_mag, GRP, 4, 3, 20, 0.01, "oracle", 42, alpha_fix, rescore=True)
    assert fit.theta.shape == fit.pairs.shape[:2]
    assert torch.equal(fit.alpha, alpha_fix)

    # joint는 순차에서 못 씀
    try:
        optimize_seq(w, x, a_mag, GRP, 4, 2, 20, 0.01, "oracle", 42, None, rescore=True)
        raise AssertionError("alpha 고정 없이 통과됨")
    except ValueError:
        pass


def test_sched_rescore():
    # rescore = seq_rescore의 pair + all의 theta 학습. 두 축이 정말 갈리는지
    x, w, a_mag = toy()
    alpha_fix = torch.ones(D_IN)

    for n_rots in (1, 4):
        seq_re = optimize_seq(w, x, a_mag, GRP, 4, n_rots, 20, 0.01, "oracle", 42, alpha_fix, rescore=True)
        got = optimize(w, x, seq_re.pairs, seq_re.mask, GRP, 4, 20, 0.01, alpha_fix)
        assert torch.equal(got.pairs, seq_re.pairs), "pair는 seq_rescore와 같아야 함"

        pairs_all, mask_all = build_pairs(w, a_mag, GRP, n_rots, "oracle", seed=42)
        want_all = optimize(w, x, pairs_all, mask_all, GRP, 4, 20, 0.01, alpha_fix)
        if n_rots == 1:
            # rot이 하나면 재채점할 것이 없어 pair가 all과 같고, theta 학습도 같으므로 loss까지 일치
            assert torch.equal(seq_re.pairs, pairs_all)
            assert abs(got.loss - want_all.loss) < 1e-9, (got.loss, want_all.loss)
        else:
            assert not torch.equal(seq_re.pairs, pairs_all), "재채점하면 뒤쪽 rot이 갈려야 함"
            # theta를 한꺼번에 학습하므로 순차로 얼려 학습한 seq_rescore와는 값이 달라짐
            assert got.loss != seq_re.loss


def test_optimize_beats_rtn():
    x, w, a_mag = toy()
    loss_rtn = (x @ (fake_quant(w, GRP, 4) - w).T).pow(2).mean().item()

    pairs, mask = build_pairs(w, a_mag, GRP, 4, "rand", seed=42)
    fit = optimize(w, x, pairs, mask, GRP, 4, n_steps=30, lr=0.01)
    assert fit.loss < loss_rtn, (fit.loss, loss_rtn)
    assert fit.theta.shape == pairs.shape[:2]

    # rot 없이 alpha만 걸어도 RTN보다는 나아야 함
    fit_scale = optimize(w, x, None, None, GRP, 4, n_steps=30, lr=0.01)
    assert fit_scale.theta is None and fit_scale.loss < loss_rtn

    # alpha를 1로 묶고 rot도 없으면 학습할 것이 없음
    try:
        optimize(w, x, None, None, GRP, 4, 30, 0.01, alpha_fix=torch.ones(D_IN))
        raise AssertionError("빈 파라미터를 막지 못함")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("전부 통과")
