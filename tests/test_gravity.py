"""tfn.gravity 的測試：論文 §5.2 的牛頓重力示範。

移植對照：reference/tensorfieldnetworks-tf/gravity.ipynb

本檔的關卡在最後兩條 slow 測試：訓練完之後，網路內部那條徑向函數要自己
長成 −1/r²。整個網路只有一條徑向函數可學，所以那條曲線對不對，就是這個
實驗成不成立。

其餘的快測試都是在保護那兩條：資料產生器、解析解、等變性任何一環錯了，
曲線都不會對，但失敗訊息會指向真正的源頭而不是「曲線不像」。
"""

import math

import numpy as np
import pytest
import torch

from tfn import gravity, utils

# 比對區間：下限是最小點距（更近的點對根本不會被生出來），上限是 RBF 中心
# 的上限。實測 73% 的點對距離超過 2.0、56% 超過 2.5，而 RBF 在 2.5 之後
# 實質失效，所以那一段的曲線沒有意義。
FIT_LOW = 0.5
FIT_HIGH = 2.0
FIT_SAMPLES = 50


# --------------------------------------------------------------------------
# 資料產生器
# --------------------------------------------------------------------------


def test_random_points_respect_the_minimum_separation():
    """剔除近距離的點是必要的：目標是 1/r²，靠太近的點對答案會爆掉。"""
    for seed in range(30):
        points, _ = gravity.random_points_and_masses(seed)
        if len(points) < 2:
            continue
        distances = torch.linalg.vector_norm(points[:, None] - points[None, :], dim=-1)
        off_diagonal = distances[~torch.eye(len(points), dtype=torch.bool)]
        assert off_diagonal.min().item() >= gravity.MIN_SEPARATION


def test_random_points_stay_inside_the_box_and_mass_range():
    points, masses = gravity.random_points_and_masses(0)
    assert points.abs().max().item() <= gravity.MAX_COORD
    assert masses.min().item() >= gravity.MASS_LOW
    assert masses.max().item() <= gravity.MASS_HIGH
    assert len(masses) == len(points)


def test_random_points_can_fall_below_two_after_filtering():
    """剔除之後可能只剩不到 2 個點，那一步沒有任何梯度。

    實測約 0.04% 的步驟會這樣。這是預期行為不是 bug，訓練迴圈必須撐得住。
    """
    counts = {len(gravity.random_points_and_masses(seed)[0]) for seed in range(200)}
    assert min(counts) >= 1
    assert max(counts) <= gravity.TRAIN_MAX_POINTS


def test_random_points_are_reproducible_from_a_seed():
    first, first_masses = gravity.random_points_and_masses(3)
    second, second_masses = gravity.random_points_and_masses(3)
    assert torch.equal(first, second)
    assert torch.equal(first_masses, second_masses)


def test_random_points_advance_a_shared_generator():
    rng = np.random.default_rng(0)
    first, _ = gravity.random_points_and_masses(rng)
    second, _ = gravity.random_points_and_masses(rng)
    assert first.shape != second.shape or not torch.equal(first, second)


# --------------------------------------------------------------------------
# 解析解
# --------------------------------------------------------------------------


def test_accelerations_match_a_hand_computed_two_body_case():
    """兩個點相距 2，質量 1 與 3。a_i = −Σ_j m_j r̂_ij / d_ij²。"""
    points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    masses = torch.tensor([1.0, 3.0])
    accel = gravity.accelerations(points, masses)
    # 點 0 被點 1 往 +x 拉：3 / 2² = 0.75
    assert accel[0].tolist() == pytest.approx([0.75, 0.0, 0.0])
    # 點 1 被點 0 往 −x 拉：1 / 2² = 0.25
    assert accel[1].tolist() == pytest.approx([-0.25, 0.0, 0.0])


def test_accelerations_obey_the_inverse_square_law():
    """距離加倍，加速度變成四分之一。"""
    near = gravity.accelerations(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), torch.tensor([1.0, 1.0])
    )
    far = gravity.accelerations(
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), torch.tensor([1.0, 1.0])
    )
    assert near[0, 0].item() == pytest.approx(4.0 * far[0, 0].item())


def test_accelerations_are_zero_for_a_single_point():
    accel = gravity.accelerations(torch.zeros(1, 3), torch.ones(1))
    assert torch.equal(accel, torch.zeros(1, 3))


def test_accelerations_are_rotation_equivariant():
    """解析解本身必須等變，否則拿它當 ground truth 就是在教網路破壞等變性。"""
    points, masses = gravity.random_points_and_masses(1)
    rotation = utils.random_rotation_matrix(5)
    rotated = gravity.accelerations(points @ rotation.T, masses)
    assert torch.allclose(rotated, gravity.accelerations(points, masses) @ rotation.T, atol=1e-5)


def test_accelerations_are_translation_invariant():
    points, masses = gravity.random_points_and_masses(1)
    shift = torch.tensor([1.5, -2.0, 0.7])
    assert torch.allclose(
        gravity.accelerations(points + shift, masses),
        gravity.accelerations(points, masses),
        atol=1e-5,
    )


# --------------------------------------------------------------------------
# 網路
# --------------------------------------------------------------------------


def test_model_outputs_one_vector_per_point():
    model = gravity.GravityModel()
    points, masses = gravity.random_points_and_masses(2)
    assert model(points, masses).shape == (len(points), 3)


def test_model_uses_glorot_biases_like_the_upstream_notebook():
    """重力這份 notebook 明確傳了 biases_initializer=glorot_uniform。

    退回全零不會報錯也不會讓 loss 難看，只會換掉一個作者刻意選過的起點——
    而這個實驗只跑 1001 步，起點是有影響的。
    """
    torch.manual_seed(0)
    model = gravity.GravityModel()
    assert model.path.filter.radial.linear1.bias.abs().max().item() > 0.0
    assert model.path.filter.radial.linear2.bias.abs().max().item() > 0.0


def test_model_is_rotation_equivariant():
    """L=1 輸出：座標轉 R，加速度就跟著轉 R。"""
    torch.manual_seed(0)
    model = gravity.GravityModel()
    points, masses = gravity.random_points_and_masses(2)
    rotation = utils.random_rotation_matrix(11)
    with torch.no_grad():
        rotated = model(points @ rotation.T, masses)
        expected = model(points, masses) @ rotation.T
    assert torch.allclose(rotated, expected, atol=1e-5)


def test_model_is_translation_invariant():
    torch.manual_seed(0)
    model = gravity.GravityModel()
    points, masses = gravity.random_points_and_masses(2)
    with torch.no_grad():
        shifted = model(points + torch.tensor([1.5, -2.0, 0.7]), masses)
        expected = model(points, masses)
    assert torch.allclose(shifted, expected, atol=1e-5)


def test_model_has_exactly_one_radial_function_to_learn():
    """這個實驗的全部賣點：可學的東西少到可以整條畫出來跟公式比。"""
    model = gravity.GravityModel()
    radial_parameters = set(model.path.filter.radial.parameters())
    assert set(model.parameters()) == radial_parameters


def test_radial_curve_has_one_value_per_distance():
    model = gravity.GravityModel()
    distances = torch.linspace(FIT_LOW, FIT_HIGH, FIT_SAMPLES)
    assert model.radial_curve(distances).shape == (FIT_SAMPLES,)


def test_analytic_radial_is_the_inverse_square_law():
    distances = torch.tensor([0.5, 1.0, 2.0])
    assert gravity.analytic_radial(distances).tolist() == pytest.approx([-4.0, -1.0, -0.25])


# --------------------------------------------------------------------------
# 訓練到收斂（slow）
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model() -> gravity.GravityModel:
    torch.manual_seed(0)
    model = gravity.GravityModel()
    gravity.train(model, steps=gravity.TRAIN_STEPS, rng=0)
    return model


@pytest.mark.slow
def test_validation_loss_converges(trained_model: gravity.GravityModel):
    loss = gravity.validation_loss(trained_model, samples=200, rng=1)
    assert loss < gravity.VALIDATION_LOSS_CEILING, f"validation loss = {loss:.3f}"


@pytest.mark.slow
def test_the_learned_radial_function_is_the_inverse_square_law(
    trained_model: gravity.GravityModel,
):
    """本票的兌現點：沒有人告訴網路 1/r²，它自己從加速度資料裡長出來。

    區間外不比對——RBF 中心只鋪到 2.0，更遠的距離網路一律吐同一個常數。
    """
    distances = torch.linspace(FIT_LOW, FIT_HIGH, FIT_SAMPLES)
    error = utils.normalized_rmse(
        trained_model.radial_curve(distances), gravity.analytic_radial(distances)
    ).item()
    assert error < gravity.RADIAL_NRMSE_CEILING, f"nRMSE = {error:.4f}"


@pytest.mark.slow
def test_the_learned_radial_function_has_the_right_sign(
    trained_model: gravity.GravityModel,
):
    """重力是吸引力，整條曲線都該是負的。

    符號整條反轉時 nRMSE 會是 2.0、上面那條也會紅，但這條的失敗訊息更直白。
    """
    distances = torch.linspace(FIT_LOW, FIT_HIGH, FIT_SAMPLES)
    assert trained_model.radial_curve(distances).max().item() < 0.0


def test_training_is_reproducible_from_a_seed():
    """同一個 seed 跑兩次要一模一樣，否則慢測試的門檻沒有意義。"""
    losses = []
    for _ in range(2):
        torch.manual_seed(0)
        model = gravity.GravityModel()
        losses.append(gravity.train(model, steps=5, rng=0))
    assert losses[0] == pytest.approx(losses[1])


def test_training_survives_a_step_with_too_few_points():
    """剔除之後只剩 1 個點的那一步沒有梯度，訓練迴圈不能因此炸掉。"""
    torch.manual_seed(0)
    model = gravity.GravityModel()
    single = torch.zeros(1, 3)
    assert math.isfinite(gravity.step_loss(model, single, torch.ones(1)).item())
