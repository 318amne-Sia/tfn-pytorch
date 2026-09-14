"""tfn.moment_of_inertia 的測試：論文 §5.2 的轉動慣量示範。

移植對照：reference/tensorfieldnetworks-tf/moment_of_inertia.ipynb

本檔的關卡是最後那兩條 slow 測試：訓練完之後，網路裡的**兩條**徑向函數要
分別長成 2/3·r² 和 −r²。那正是轉動慣量公式拆成 L=0 與 L=2 兩部分之後的
係數——網路沒有別的自由度可以吸收它們，所以曲線的尺度是絕對釘死的。

其餘快測試都在保護那兩條：資料產生器、解析解、等變性任何一環錯了曲線都不會
對，但失敗訊息會指向真正的源頭。
"""

import numpy as np
import pytest
import torch

from tfn import moment_of_inertia as moi
from tfn import utils

FIT_SAMPLES = 50


# --------------------------------------------------------------------------
# 資料產生器
# --------------------------------------------------------------------------


def test_random_points_have_a_fixed_count_and_range():
    points, masses = moi.random_points_and_masses(0)
    assert points.shape == (moi.NUM_POINTS, 3)
    assert masses.shape == (moi.NUM_POINTS,)
    assert points.abs().max().item() <= moi.MAX_COORD


def test_the_centre_point_has_zero_mass():
    """L=0 濾波器沒有「自己對自己」的遮罩，中心點的質量會經由 R₀(距離≈0)
    直接漏進輸出。解析解那邊它本來就貢獻 0（相對位置是零向量），所以歸零
    只動到網路那一側。

    這是照抄原作的作法。另一條路是給 filter_0 也加對角線遮罩，但那會動到
    實驗一也在用的共用程式碼。
    """
    _, masses = moi.random_points_and_masses(0)
    assert masses[moi.CENTRE_INDEX].item() == 0.0
    assert masses[1:].min().item() >= moi.MASS_LOW
    assert masses[1:].max().item() <= moi.MASS_HIGH


def test_random_points_are_reproducible_from_a_seed():
    first = moi.random_points_and_masses(3)
    second = moi.random_points_and_masses(3)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_random_points_advance_a_shared_generator():
    rng = np.random.default_rng(0)
    assert not torch.equal(
        moi.random_points_and_masses(rng)[0], moi.random_points_and_masses(rng)[0]
    )


# --------------------------------------------------------------------------
# 解析解
# --------------------------------------------------------------------------


def test_moment_of_inertia_matches_a_hand_computed_case():
    """中心在原點，一個質量 3 的點在 (2, 0, 0)。

    I = m(|r|²δ − r r) → 繞 x 軸轉不費力（Ixx = 0），繞 y / z 軸是 m·d² = 12。
    """
    points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    masses = torch.tensor([0.0, 3.0])
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]])
    assert torch.allclose(moi.moment_of_inertia(points, masses, index=0), expected, atol=1e-6)


def test_moment_of_inertia_is_symmetric():
    points, masses = moi.random_points_and_masses(1)
    inertia = moi.moment_of_inertia(points, masses)
    assert torch.allclose(inertia, inertia.T, atol=1e-6)


def test_moment_of_inertia_is_measured_about_the_chosen_centre():
    """換一個中心就是另一個答案——它不是點雲的全域屬性。"""
    points, masses = moi.random_points_and_masses(1)
    assert not torch.allclose(
        moi.moment_of_inertia(points, masses, index=0),
        moi.moment_of_inertia(points, masses, index=1),
        atol=1e-3,
    )


def test_moment_of_inertia_rotates_as_a_matrix():
    """解析解自己必須等變，否則就是拿一組會破壞等變性的答案去教網路。"""
    points, masses = moi.random_points_and_masses(1)
    rotation = utils.random_rotation_matrix(5)
    rotated = moi.moment_of_inertia(points @ rotation.T, masses)
    expected = rotation @ moi.moment_of_inertia(points, masses) @ rotation.T
    assert torch.allclose(rotated, expected, atol=1e-5)


def test_moment_of_inertia_is_translation_invariant():
    """位置都是相對於中心點算的，整體平移看不見。"""
    points, masses = moi.random_points_and_masses(1)
    shift = torch.tensor([1.5, -2.0, 0.7])
    assert torch.allclose(
        moi.moment_of_inertia(points + shift, masses),
        moi.moment_of_inertia(points, masses),
        atol=1e-5,
    )


# --------------------------------------------------------------------------
# 網路
# --------------------------------------------------------------------------


def test_model_outputs_a_matrix_per_point():
    """卷積天生每個點都算一份；loss 只取第 0 個，其餘丟掉。"""
    model = moi.MomentOfInertiaModel()
    points, masses = moi.random_points_and_masses(2)
    assert model(points, masses).shape == (moi.NUM_POINTS, 3, 3)


def test_model_output_is_symmetric():
    """對稱是 matrix_from_0_2 的結構保證，不是訓練出來的。"""
    torch.manual_seed(0)
    model = moi.MomentOfInertiaModel()
    points, masses = moi.random_points_and_masses(2)
    with torch.no_grad():
        out = model(points, masses)
    assert torch.equal(out, out.transpose(-2, -1))


def test_model_rotates_as_a_matrix():
    """0 → 0 ⊕ 2 的等變性，端對端測在拼好的 3×3 上。"""
    torch.manual_seed(0)
    model = moi.MomentOfInertiaModel()
    points, masses = moi.random_points_and_masses(2)
    rotation = utils.random_rotation_matrix(11)
    with torch.no_grad():
        rotated = model(points @ rotation.T, masses)
        expected = rotation @ model(points, masses) @ rotation.T
    assert torch.allclose(rotated, expected, atol=1e-5)


def test_model_is_translation_invariant():
    torch.manual_seed(0)
    model = moi.MomentOfInertiaModel()
    points, masses = moi.random_points_and_masses(2)
    with torch.no_grad():
        shifted = model(points + torch.tensor([1.5, -2.0, 0.7]), masses)
        expected = model(points, masses)
    assert torch.allclose(shifted, expected, atol=1e-5)


def test_model_has_exactly_two_radial_functions_to_learn():
    """整個實驗的賣點：可學的東西少到兩條曲線都能攤開來跟公式比。"""
    model = moi.MomentOfInertiaModel()
    radial = set(model.path_0.filter.radial.parameters()) | set(
        model.path_2.filter.radial.parameters()
    )
    assert set(model.parameters()) == radial


def test_model_uses_zero_biases_like_the_upstream_notebook():
    """與票 09 的重力相反：那份 notebook 明確傳了 glorot，這份吃預設的全零。"""
    model = moi.MomentOfInertiaModel()
    for path in (model.path_0, model.path_2):
        bias = path.filter.radial.linear1.bias
        assert torch.equal(bias, torch.zeros_like(bias))


def test_radial_curves_give_one_value_per_distance_for_each_path():
    model = moi.MomentOfInertiaModel()
    distances = torch.linspace(moi.FIT_LOW, moi.FIT_HIGH, FIT_SAMPLES)
    l0, l2 = model.radial_curves(distances)
    assert l0.shape == (FIT_SAMPLES,)
    assert l2.shape == (FIT_SAMPLES,)


def test_analytic_radials_are_the_two_halves_of_the_formula():
    distances = torch.tensor([0.5, 1.0, 2.0])
    assert moi.analytic_radial_0(distances).tolist() == pytest.approx([1 / 6, 2 / 3, 8 / 3])
    assert moi.analytic_radial_2(distances).tolist() == pytest.approx([-0.25, -1.0, -4.0])


def test_the_analytic_radials_reproduce_the_formula_exactly():
    """把解析的徑向函數手動代進網路的算式，應該得到解析的轉動慣量張量。

    這條是整個實驗的數學前提：轉動慣量剛好寫得成 TFN 的形式。它若不成立，
    網路再怎麼訓練也對不上，而且是設計錯不是訓練錯。
    """
    points, masses = moi.random_points_and_masses(4)
    centre = points[moi.CENTRE_INDEX]
    rij = points - centre
    distances = torch.linalg.vector_norm(rij, dim=-1)

    scalar = (moi.analytic_radial_0(distances) * masses).sum()
    from tfn.layers import Y_2, matrix_from_0_2

    l2 = (moi.analytic_radial_2(distances) * masses).unsqueeze(-1) * Y_2(rij)
    built = matrix_from_0_2(scalar.reshape(()), l2.sum(dim=0))

    assert torch.allclose(built, moi.moment_of_inertia(points, masses), atol=1e-5)


# --------------------------------------------------------------------------
# 訓練到收斂（slow）
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model() -> moi.MomentOfInertiaModel:
    torch.manual_seed(0)
    model = moi.MomentOfInertiaModel()
    moi.train(model, steps=moi.TRAIN_STEPS, rng=0)
    return model


@pytest.mark.slow
def test_validation_loss_converges(trained_model: moi.MomentOfInertiaModel):
    loss = moi.validation_loss(trained_model, samples=200, rng=1)
    assert loss < moi.VALIDATION_LOSS_CEILING, f"validation loss = {loss:.5f}"


@pytest.mark.slow
def test_the_learned_radial_functions_match_the_analytic_solution(
    trained_model: moi.MomentOfInertiaModel,
):
    """本票的兌現點。沒有人告訴網路 2/3·r² 與 −r²，它從轉動慣量張量裡自己長出來。"""
    distances = torch.linspace(moi.FIT_LOW, moi.FIT_HIGH, FIT_SAMPLES)
    learned_0, learned_2 = trained_model.radial_curves(distances)

    error_0 = utils.normalized_rmse(learned_0, moi.analytic_radial_0(distances)).item()
    error_2 = utils.normalized_rmse(learned_2, moi.analytic_radial_2(distances)).item()

    assert error_0 < moi.RADIAL_L0_NRMSE_CEILING, f"L=0 的 nRMSE = {error_0:.4f}"
    assert error_2 < moi.RADIAL_L2_NRMSE_CEILING, f"L=2 的 nRMSE = {error_2:.4f}"


@pytest.mark.slow
def test_the_learned_radial_functions_have_the_right_signs(
    trained_model: moi.MomentOfInertiaModel,
):
    """L=0 那條整條為正、L=2 那條整條為負。

    符號整條反轉時 nRMSE 會是 2.0、上面那條也會紅，但這條的訊息更直白。
    """
    distances = torch.linspace(moi.FIT_LOW, moi.FIT_HIGH, FIT_SAMPLES)
    learned_0, learned_2 = trained_model.radial_curves(distances)
    assert learned_0.min().item() > 0.0
    assert learned_2.max().item() < 0.0


def test_training_is_reproducible_from_a_seed():
    losses = []
    for _ in range(2):
        torch.manual_seed(0)
        model = moi.MomentOfInertiaModel()
        losses.append(moi.train(model, steps=5, rng=0))
    assert losses[0] == pytest.approx(losses[1])
