"""tfn.utils 的測試。

移植對照：reference/tensorfieldnetworks-tf/tensorfieldnetworks/utils.py

這裡的重點是 random_rotation_matrix——票 03 之後每一張的等變性測試
都站在它上面，它若不是真正的旋轉矩陣，後面所有「等變性通過」都不算數。
"""

import math

import numpy as np
import pytest
import torch

from tfn import utils
from tfn.utils import EPSILON

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")


# --------------------------------------------------------------------------
# Levi-Civita
# --------------------------------------------------------------------------


def test_eijk_shape_and_dtype():
    e = utils.get_eijk()
    assert e.shape == (3, 3, 3)
    assert e.dtype == utils.FLOAT_TYPE


def test_eijk_is_fully_antisymmetric():
    """任兩指標交換就變號。這是 Levi-Civita 的定義性質。"""
    e = utils.get_eijk()
    assert torch.equal(e, -e.permute(1, 0, 2))  # 交換 i, j
    assert torch.equal(e, -e.permute(0, 2, 1))  # 交換 j, k
    assert torch.equal(e, -e.permute(2, 1, 0))  # 交換 i, k


def test_eijk_vanishes_on_repeated_indices():
    e = utils.get_eijk()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if len({i, j, k}) < 3:
                    assert e[i, j, k] == 0.0, f"ε[{i},{j},{k}] 應為 0"


def test_eijk_sign_convention():
    """偶排列 +1、奇排列 -1。符號弄反的話票 04 的外積會整個反向。"""
    e = utils.get_eijk()
    assert e[0, 1, 2] == 1.0
    assert e[1, 2, 0] == 1.0
    assert e[2, 0, 1] == 1.0
    assert e[0, 2, 1] == -1.0
    assert e[2, 1, 0] == -1.0
    assert e[1, 0, 2] == -1.0


def test_eijk_reproduces_cross_product():
    """ε 的用途就是外積：(a × b)_i = ε_ijk a_j b_k。

    票 04 的 1×1→1 路徑靠這個關係，這裡先把它釘死。
    """
    e = utils.get_eijk()
    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, generator=g)
    b = torch.randn(3, generator=g)
    assert torch.allclose(torch.einsum("ijk,j,k->i", e, a, b), torch.linalg.cross(a, b), atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_eijk_respects_device(device):
    assert utils.get_eijk(device=device).device.type == device


# --------------------------------------------------------------------------
# norm_with_epsilon
# --------------------------------------------------------------------------


def test_norm_with_epsilon_matches_plain_norm_away_from_zero():
    x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    assert torch.allclose(utils.norm_with_epsilon(x, dim=-1), torch.tensor([5.0, 13.0]))


def test_norm_with_epsilon_floors_at_sqrt_epsilon():
    """零向量不會回 0，而是 sqrt(EPSILON)。

    這是刻意的：後面要拿它當分母算單位向量。
    """
    got = utils.norm_with_epsilon(torch.zeros(3))
    assert got.item() == pytest.approx(math.sqrt(EPSILON), rel=1e-6)


def test_norm_with_epsilon_has_no_nan_gradient_at_zero():
    """在零點反向傳播不能出 NaN——這正是原版用 max 而不是加 epsilon 的原因。"""
    x = torch.zeros(3, requires_grad=True)
    utils.norm_with_epsilon(x).backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_norm_with_epsilon_keepdim():
    x = torch.ones(2, 3, 4)
    assert utils.norm_with_epsilon(x, dim=-1).shape == (2, 3)
    assert utils.norm_with_epsilon(x, dim=-1, keepdim=True).shape == (2, 3, 1)


# --------------------------------------------------------------------------
# shifted softplus
# --------------------------------------------------------------------------


def test_ssp_matches_naive_formula_in_safe_range():
    """與原版 log(0.5·exp(x)+0.5) 在不溢位的範圍內數值相同。"""
    x = torch.linspace(-20.0, 20.0, 201)
    assert torch.allclose(utils.ssp(x), torch.log(0.5 * torch.exp(x) + 0.5), atol=1e-6)


def test_ssp_passes_through_origin():
    """「shifted」的意思：減掉 log2 讓它過原點，一般 softplus(0) = 0.693。"""
    assert utils.ssp(torch.zeros(1)).item() == pytest.approx(0.0, abs=1e-7)


def test_ssp_does_not_overflow_where_the_naive_form_does():
    x = torch.tensor([100.0])
    assert torch.isinf(torch.log(0.5 * torch.exp(x) + 0.5)).all()  # 原版寫法在此爆掉
    assert torch.isfinite(utils.ssp(x)).all()
    assert utils.ssp(x).item() == pytest.approx(100.0 - math.log(2.0), rel=1e-6)


# --------------------------------------------------------------------------
# 幾何矩陣
# --------------------------------------------------------------------------


def test_difference_matrix_index_order():
    """rij[i, j] = r_i - r_j。順序弄反會讓所有 L=1 特徵反向。"""
    r = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    rij = utils.difference_matrix(r)
    assert rij.shape == (2, 2, 3)
    assert torch.equal(rij[0, 1], r[0] - r[1])
    assert torch.equal(rij[1, 0], r[1] - r[0])


def test_difference_matrix_diagonal_is_exactly_zero():
    r = torch.randn(5, 3)
    rij = utils.difference_matrix(r)
    assert torch.equal(rij[range(5), range(5)], torch.zeros(5, 3))


def test_difference_matrix_is_antisymmetric():
    r = torch.randn(5, 3)
    rij = utils.difference_matrix(r)
    assert torch.allclose(rij, -rij.transpose(0, 1))


def test_distance_matrix_matches_independent_computation():
    """非對角元素要與獨立算法一致。

    對角線刻意不比——它被 EPSILON 夾住，由下面那條測試單獨守。
    """
    r = torch.randn(6, 3)
    dij = utils.distance_matrix(r)
    assert dij.shape == (6, 6)
    off_diagonal = ~torch.eye(6, dtype=torch.bool)
    assert torch.allclose(dij[off_diagonal], torch.cdist(r, r)[off_diagonal], atol=1e-5)


def test_distance_matrix_diagonal_is_the_epsilon_floor_not_zero():
    """對角線是 sqrt(EPSILON) = 1e-4，不是 0。

    因為 distance_matrix 走的是 norm_with_epsilon。票 03 的 F_1 遮罩不能
    用這個值去比 `< EPSILON`（1e-4 < 1e-8 是 False，遮罩不會生效），
    上游那裡用的是未正則化的 norm。
    """
    dij = utils.distance_matrix(torch.randn(4, 3))
    diag = dij[range(4), range(4)]
    assert torch.allclose(diag, torch.full((4,), math.sqrt(EPSILON)), rtol=1e-5)
    assert (diag > EPSILON).all()


def test_distance_matrix_is_symmetric():
    dij = utils.distance_matrix(torch.randn(5, 3))
    assert torch.allclose(dij, dij.T)


@pytest.mark.parametrize("device", DEVICES)
def test_geometry_helpers_follow_input_device(device):
    r = torch.randn(4, 3, device=device)
    assert utils.difference_matrix(r).device.type == device
    assert utils.distance_matrix(r).device.type == device


# --------------------------------------------------------------------------
# RBF 展開
# --------------------------------------------------------------------------


def test_rbf_expansion_adds_one_axis_per_centre():
    d = utils.distance_matrix(torch.randn(5, 3))
    assert utils.rbf_expansion(d, low=0.0, high=3.5, count=4).shape == (5, 5, 4)


def test_rbf_expansion_peaks_at_its_centres():
    """距離剛好落在某個中心上時，那一格的響應是 1，其餘都更小。"""
    centers = torch.linspace(0.0, 3.5, 4)
    got = utils.rbf_expansion(centers, low=0.0, high=3.5, count=4)
    for i, row in enumerate(got):
        assert row[i].item() == pytest.approx(1.0, abs=1e-6)
        assert row.argmax().item() == i


def test_rbf_expansion_pins_both_upstream_conventions():
    """對著手寫的常數釘死，而不是把實作的式子再抄一遍。

    兩個會弄錯的地方，這裡各寫成字面值：

    - 中心含兩端（``linspace(low, high, count)``），(0, 3.5, 4) 就是
      ``[0, 7/6, 7/3, 3.5]``，不是 ``[0, 0.875, 1.75, 2.625]``
    - 寬度的分母是 ``count`` 而不是 ``count - 1``，gamma = 1 / 0.875
    """
    got = utils.rbf_expansion(torch.tensor([1.0]), low=0.0, high=3.5, count=4)[0]
    expected = [math.exp(-((1.0 - centre) ** 2) / 0.875) for centre in (0.0, 7 / 6, 7 / 3, 3.5)]
    assert got.tolist() == pytest.approx(expected, rel=1e-6)


def test_rbf_expansion_promotes_integer_distances():
    """整數距離不能讓中心被截成 [0, 1, 2, 3]——那會算出位置錯掉的響應。"""
    integer = utils.rbf_expansion(torch.arange(3), low=0.0, high=3.5, count=4)
    float_ = utils.rbf_expansion(torch.arange(3).float(), low=0.0, high=3.5, count=4)
    assert integer.dtype.is_floating_point
    assert torch.allclose(integer, float_)


@pytest.mark.parametrize("device", DEVICES)
def test_rbf_expansion_follows_input_device(device):
    d = utils.distance_matrix(torch.randn(4, 3, device=device))
    assert utils.rbf_expansion(d, low=0.0, high=3.5, count=4).device.type == device


# --------------------------------------------------------------------------
# 旋轉矩陣
# --------------------------------------------------------------------------


def test_rotation_matrix_is_orthogonal_with_unit_determinant():
    r = utils.rotation_matrix(np.array([0.3, -0.5, 0.8]), 1.234)
    assert r.shape == (3, 3)
    assert torch.allclose(r.T @ r, torch.eye(3), atol=1e-6)
    assert torch.linalg.det(r).item() == pytest.approx(1.0, abs=1e-6)


def test_rotation_matrix_convention_is_right_handed():
    """R @ v 是把 v 繞 axis 依右手定則轉 theta。

    繞 z 軸轉 90 度，x 軸應該轉到 y 軸。轉錯方向（拿到 R 的轉置）
    會讓等變性測試變成在驗一個錯的關係，卻照樣通過。
    """
    r = utils.rotation_matrix(np.array([0.0, 0.0, 1.0]), math.pi / 2)
    x_axis = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(r @ x_axis, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_rotation_matrix_leaves_its_axis_fixed():
    axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    r = utils.rotation_matrix(axis, 0.7)
    a = torch.as_tensor(axis, dtype=utils.FLOAT_TYPE)
    assert torch.allclose(r @ a, a, atol=1e-6)


def test_rotation_matrix_zero_angle_is_identity():
    r = utils.rotation_matrix(np.array([0.2, 0.9, -0.3]), 0.0)
    assert torch.allclose(r, torch.eye(3), atol=1e-6)


def test_random_rotation_matrix_is_a_proper_rotation():
    for seed in range(20):
        r = utils.random_rotation_matrix(seed)
        assert torch.allclose(r.T @ r, torch.eye(3), atol=1e-5), f"seed {seed} 不正交"
        assert torch.linalg.det(r).item() == pytest.approx(1.0, abs=1e-5), f"seed {seed} det≠1"


def test_random_rotation_matrix_is_reproducible_from_a_seed():
    assert torch.equal(utils.random_rotation_matrix(42), utils.random_rotation_matrix(42))


def test_random_rotation_matrix_differs_across_seeds():
    assert not torch.allclose(utils.random_rotation_matrix(1), utils.random_rotation_matrix(2))


def test_random_rotation_matrix_accepts_a_generator():
    """傳同一個 Generator 應該連續產出不同的旋轉，而不是每次重置。"""
    rng = np.random.default_rng(0)
    first = utils.random_rotation_matrix(rng)
    second = utils.random_rotation_matrix(rng)
    assert not torch.allclose(first, second)
    assert torch.equal(first, utils.random_rotation_matrix(np.random.default_rng(0)))


@pytest.mark.parametrize("device", DEVICES)
def test_rotation_matrix_respects_device(device):
    assert utils.random_rotation_matrix(0, device=device).device.type == device


def test_rotation_preserves_distances():
    """旋轉是保距的——這是等變性測試能成立的前提。"""
    r = torch.randn(6, 3)
    rot = utils.random_rotation_matrix(3)
    assert torch.allclose(utils.distance_matrix(r @ rot.T), utils.distance_matrix(r), atol=1e-5)
