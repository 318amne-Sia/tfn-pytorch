"""tfn.layers 的濾波器測試：徑向部分 R 與角度部分 F_0 / F_1。

移植對照：reference/tensorfieldnetworks-tf/tensorfieldnetworks/layers.py

本檔的關卡是等變性：F_0 的輸出在旋轉下不變、F_1 的輸出跟著轉。
兩者都靠票 02 的 random_rotation_matrix，那支已在 test_utils 驗過是真旋轉。
"""

import math
from typing import cast

import pytest
import torch
from torch.nn import functional as F

from tfn import layers, utils

# 這裡只需要「某一組」RBF 設定；沿用實驗一的值純粹是方便跟 reference 對照。
# 刻意不從 tfn.shape_classification import——濾波器的測試不該綁在某個實驗上，
# 這幾個數字改掉也不會影響本檔任何一條斷言的意義。
RBF_LOW = 0.0
RBF_HIGH = 3.5
RBF_COUNT = 4


def rbf_expansion(geometry: torch.Tensor) -> torch.Tensor:
    """``[N, 3] -> [N, N, RBF_COUNT]``，高斯基底展開的點對距離。

    距離是旋轉不變也是平移不變的，所以 rbf 在等變性測試裡是常數——
    F_1 輸出會轉，靠的全是角度部分。
    """
    return utils.rbf_expansion(
        utils.distance_matrix(geometry), low=RBF_LOW, high=RBF_HIGH, count=RBF_COUNT
    )


def geometry(n: int = 5, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=g)


# --------------------------------------------------------------------------
# R：徑向函數
# --------------------------------------------------------------------------


def test_R_maps_rbf_to_one_weight_per_pair_and_channel():
    radial = layers.R(RBF_COUNT, output_dim=5)
    assert radial(rbf_expansion(geometry(6))).shape == (6, 6, 5)


def test_R_hidden_dim_defaults_to_input_dim():
    """同作者：hidden_dim 未指定時等於輸入維度。"""
    radial = layers.R(7)
    assert radial.linear1.out_features == 7
    assert radial.linear2.in_features == 7


def test_R_hidden_dim_can_be_given_explicitly():
    radial = layers.R(7, hidden_dim=13)
    assert radial.linear1.out_features == 13
    assert radial.linear2.in_features == 13


def test_R_initialises_biases_to_zero():
    """對齊作者的 constant_initializer(0.)。

    nn.Linear 預設的 bias 是均勻亂數，這條會在忘記初始化時直接紅。
    """
    radial = layers.R(4, output_dim=3)
    assert torch.equal(radial.linear1.bias, torch.zeros_like(radial.linear1.bias))
    assert torch.equal(radial.linear2.bias, torch.zeros_like(radial.linear2.bias))


def test_R_initialises_weights_with_xavier_uniform_not_the_torch_default():
    """對齊作者的 xavier_initializer，而不是 nn.Linear 預設的 Kaiming uniform。

    兩者都是均勻分布，差在界寬：fan_in = fan_out = d 時 Xavier 的界是
    sqrt(3/d)、PyTorch 預設是 sqrt(1/d)，差 sqrt(3) 倍。取夠大的 d 讓
    抽樣誤差遠小於這個差距，就能把兩者分開。
    """
    torch.manual_seed(0)
    dim = 256
    radial = layers.R(dim, output_dim=dim)
    xavier_bound = math.sqrt(6.0 / (dim + dim))
    torch_default_bound = math.sqrt(1.0 / dim)
    for linear in (radial.linear1, radial.linear2):
        largest = linear.weight.abs().max().item()
        assert largest <= xavier_bound
        assert largest > torch_default_bound, "看起來還是 nn.Linear 的預設初始化"
        # 均勻分布 U(-b, b) 的標準差是 b/sqrt(3)
        assert linear.weight.std().item() == pytest.approx(xavier_bound / math.sqrt(3.0), rel=0.05)


def test_R_bias_init_zeros_is_explicit_as_well_as_default():
    """明確傳 "zeros" 與不傳要一樣——實驗一整個吃這個預設。"""
    radial = layers.R(4, output_dim=3, bias_init="zeros")
    assert torch.equal(radial.linear1.bias, torch.zeros_like(radial.linear1.bias))
    assert torch.equal(radial.linear2.bias, torch.zeros_like(radial.linear2.bias))


def test_R_glorot_bias_uses_each_biases_own_length():
    """對齊上游重力 notebook 的 biases_initializer=glorot_uniform。

    TF1 的 _compute_fans 對 1-D 張量取 fan_in = fan_out = shape[0]，界是
    sqrt(6 / 2n)——所以同一個 R 裡兩顆 bias 的界並不一樣寬。這裡刻意讓
    hidden_dim 與 output_dim 差很多，才擋得住「用了某個固定的 n」的寫法。

    PyTorch 的 nn.init.xavier_uniform_ 對 1-D 張量直接拋錯，界要自己算。
    """
    torch.manual_seed(0)
    radial = layers.R(4, output_dim=1024, hidden_dim=256, bias_init="glorot")

    for bias in (radial.linear1.bias, radial.linear2.bias):
        n = bias.shape[0]
        bound = math.sqrt(6.0 / (n + n))
        largest = bias.abs().max().item()
        assert largest <= bound
        assert largest > 0.9 * bound, "界看起來比 glorot 窄"

    # 分布只在夠長的那顆上驗：n = 1024 時樣本標準差的相對誤差約 2%
    b2 = radial.linear2.bias
    bound2 = math.sqrt(6.0 / (2 * b2.shape[0]))
    assert b2.std().item() == pytest.approx(bound2 / math.sqrt(3.0), rel=0.15)


def test_R_glorot_bias_leaves_the_weights_on_xavier():
    """換掉 bias 的初始化不該連帶動到權重。"""
    torch.manual_seed(0)
    dim = 256
    radial = layers.R(dim, output_dim=dim, bias_init="glorot")
    xavier_bound = math.sqrt(6.0 / (dim + dim))
    for linear in (radial.linear1, radial.linear2):
        assert linear.weight.abs().max().item() <= xavier_bound
        assert linear.weight.std().item() == pytest.approx(xavier_bound / math.sqrt(3.0), rel=0.05)


def test_R_rejects_an_unknown_bias_init():
    with pytest.raises(ValueError):
        layers.R(4, bias_init="kaiming")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "cls", [layers.filter_0, layers.filter_1_output_0, layers.filter_1_output_1]
)
def test_filter_paths_thread_bias_init_down_to_R(cls):
    """票 09 的重力實驗要靠這條路把 glorot 一路傳到最底層的 R。

    中間隔了 F_0 / F_1 兩層轉手，少接一段不會報錯、只會靜靜退回全零。
    """
    torch.manual_seed(0)
    path = cls(RBF_COUNT, bias_init="glorot")
    assert path.filter.radial.linear1.bias.abs().max().item() > 0.0


def test_filter_paths_keep_zero_bias_by_default():
    """實驗一完全吃這個預設，不能因為多了選項就改掉。"""
    path = layers.filter_0(RBF_COUNT)
    assert torch.equal(
        path.filter.radial.linear1.bias, torch.zeros_like(path.filter.radial.linear1.bias)
    )


# --- 徑向探針：把訓練好的 R 攤開成一條曲線 --------------------------------


def test_probe_radial_is_the_rbf_expansion_followed_by_R():
    """探針不是第二套實作，就是「展開再前向」——只是把 RBF 設定收在一處。"""
    torch.manual_seed(0)
    radial = layers.R(RBF_COUNT, output_dim=2)
    distances = torch.linspace(0.1, 3.0, 17)
    expected = radial(utils.rbf_expansion(distances, low=RBF_LOW, high=RBF_HIGH, count=RBF_COUNT))
    probed = layers.probe_radial(radial, distances, low=RBF_LOW, high=RBF_HIGH, count=RBF_COUNT)
    assert torch.allclose(probed, expected)


def test_probe_radial_keeps_the_channel_axis():
    radial = layers.R(RBF_COUNT, output_dim=3)
    probed = layers.probe_radial(
        radial, torch.linspace(0.1, 3.0, 11), low=RBF_LOW, high=RBF_HIGH, count=RBF_COUNT
    )
    assert probed.shape == (11, 3)


def test_probe_radial_does_not_track_gradients():
    """它是拿來看的，不是訓練路徑的一部分——回傳值要能直接餵給畫圖。"""
    radial = layers.R(RBF_COUNT)
    probed = layers.probe_radial(
        radial, torch.linspace(0.1, 3.0, 5), low=RBF_LOW, high=RBF_HIGH, count=RBF_COUNT
    )
    assert not probed.requires_grad


# --------------------------------------------------------------------------
# Y_2 與 matrix_from_0_2：L=2 的角度部分與矩陣拼裝
# --------------------------------------------------------------------------
#
# 這兩個函式沒有任何參數，全部是固定常數，而且合起來有一個封閉形式：
#
#     matrix_from_0_2(0, Y_2(r)) == r̂ r̂ᵀ − I/3
#
# 係數寫錯不會讓 loss 變難看、也不會讓曲線變得不像曲線，只會讓它安靜地偏掉。
# 訓練抓不到、靜態工具抓不到、連等變性測試都抓不到（錯的係數照樣可能等變），
# 只有這條恆等式抓得到。對照物 r̂r̂ᵀ − I/3 是三行程式，寫不錯。
#
# 也因為有它，這裡**不需要 L=2 的 Wigner D 矩陣**：等變性是這條恆等式的直接
# 推論（r̂r̂ᵀ − I/3 顯然滿足 R M Rᵀ）。自己寫一個 5×5 的 Wigner D 只是引入
# 第二個同樣容易寫錯、卻沒有獨立對照來源的東西。

# 高精度那組把恆等式釘到機器精度；float32 那組確認執行期的實際 dtype 也成立。
EXACT_DTYPES = [(torch.float64, 1e-12), (torch.float32, 1e-6)]


def directions(n: int = 200, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=g, dtype=torch.float64)


def test_Y_2_has_five_components():
    assert layers.Y_2(directions(7)).shape == (7, 5)


def test_Y_2_keeps_leading_axes():
    """上游只餵 [N, N, 3]，但票 11 的濾波器也是這個形狀，別綁死成 2 軸。"""
    assert layers.Y_2(torch.randn(4, 6, 3)).shape == (4, 6, 5)


def test_Y_2_matches_the_upstream_coefficients():
    """逐個數字對上游 layers.Y_2。順序是 xy, yz, z², zx, x²−y²。

    r = (1, 2, 3)、r² = 14：
      xy/r²                    = 2/14
      yz/r²                    = 6/14
      (−x²−y²+2z²)/(2√3·r²)    = 13/(2√3·14)
      zx/r²                    = 3/14
      (x²−y²)/(2r²)            = −3/28
    """
    out = layers.Y_2(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert out.tolist() == pytest.approx(
        [2 / 14, 6 / 14, 13 / (2 * math.sqrt(3) * 14), 3 / 14, -3 / 28]
    )


def test_Y_2_only_depends_on_direction():
    """二階球諧是零次齊次的：整體縮放距離不改變角度部分。

    這條同時擋住「忘記除以 r²」——那會讓輸出隨長度平方放大。
    """
    r = directions(50)
    assert torch.allclose(layers.Y_2(r * 7.5), layers.Y_2(r), atol=1e-12, rtol=0)


def test_Y_2_is_zero_at_the_origin_without_nan():
    """重合的兩點沒有方向可言。r² 被夾在 EPSILON 以上，所以是 0 不是 NaN。"""
    out = layers.Y_2(torch.zeros(3))
    assert torch.equal(out, torch.zeros(5))


def test_matrix_from_0_2_is_symmetric():
    """三個非對角線各只寫一次、上下三角共用，所以對稱是結構保證的。"""
    built = layers.matrix_from_0_2(torch.randn(20), torch.randn(20, 5))
    assert torch.equal(built, built.transpose(-2, -1))


def test_matrix_from_0_2_keeps_leading_axes():
    """票 12 的網路輸出帶一根通道軸，不要逼呼叫端先攤平。"""
    assert layers.matrix_from_0_2(torch.randn(4, 2), torch.randn(4, 2, 5)).shape == (4, 2, 3, 3)


def test_matrix_from_0_2_rejects_a_wrong_component_count():
    with pytest.raises(ValueError):
        layers.matrix_from_0_2(torch.randn(4), torch.randn(4, 3))


def test_matrix_from_0_2_rejects_mismatched_leading_shapes():
    """最典型的誤用：L=2 那半忘了 squeeze 掉通道軸。"""
    with pytest.raises(ValueError):
        layers.matrix_from_0_2(torch.randn(4), torch.randn(4, 1, 5))


def test_matrix_from_0_2_trace_comes_only_from_the_scalar():
    """恆等式二：trace 恆等於 3s，與 L=2 那五個數字完全無關。

    這正是「1 + 5」這個拆法成立的前提——L=2 那半必須對 trace 沒有影響力，
    否則兩半就不是各自獨立的表示。d 取隨機值，所以它證的是「對任何 d」。
    """
    g = torch.Generator().manual_seed(1)
    scalar = torch.randn(200, generator=g, dtype=torch.float64)
    l2 = torch.randn(200, 5, generator=g, dtype=torch.float64)
    traces = layers.matrix_from_0_2(scalar, l2).diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(traces, 3.0 * scalar, atol=1e-12, rtol=0)


@pytest.mark.parametrize(("dtype", "tolerance"), EXACT_DTYPES)
def test_matrix_from_0_2_of_Y_2_is_the_traceless_outer_product(dtype, tolerance):
    """恆等式一，本票的核心關卡。

    把純量設成 0、餵進某個方向的二階球諧，拼出來的矩陣精確等於
    ``r̂ r̂ᵀ − I/3``——也就是「這個方向的外積，扣掉 trace」。

    Y_2 的五個係數與 matrix_from_0_2 的每個 1/√3 只要有一個寫錯，這條就紅。
    容忍度刻意開得很緊（float64 到 1e-12）：放鬆它等於放棄這一票最大的價值。
    """
    r = directions().to(dtype)
    built = layers.matrix_from_0_2(torch.zeros(len(r), dtype=dtype), layers.Y_2(r))

    unit = r / torch.linalg.vector_norm(r, dim=-1, keepdim=True)
    expected = unit.unsqueeze(-1) * unit.unsqueeze(-2) - torch.eye(3, dtype=dtype) / 3

    assert torch.allclose(built, expected, atol=tolerance, rtol=0)


def test_matrix_from_0_2_of_Y_2_rotates_as_a_matrix():
    """L=2 的等變性，測在拼好的 3×3 上：``M' = R M Rᵀ``。

    這就是不寫 Wigner D 的原因——同一件事，用大家都會的形式表達。
    """
    r = directions(100)
    rotation = utils.random_rotation_matrix(3, dtype=torch.float64)
    zeros = torch.zeros(len(r), dtype=torch.float64)

    before = layers.matrix_from_0_2(zeros, layers.Y_2(r))
    after = layers.matrix_from_0_2(zeros, layers.Y_2(r @ rotation.T))

    assert torch.allclose(after, rotation @ before @ rotation.T, atol=1e-12, rtol=0)


def test_matrix_from_0_2_scalar_part_is_a_multiple_of_the_identity():
    """純量那半只會等量加到三個對角線上——它管的是「整體大小」。"""
    l2 = torch.zeros(6, 5, dtype=torch.float64)
    scalar = torch.randn(6, dtype=torch.float64)
    built = layers.matrix_from_0_2(scalar, l2)
    assert torch.allclose(built, scalar[:, None, None] * torch.eye(3, dtype=torch.float64))


# --------------------------------------------------------------------------
# F_0：L = 0 濾波器
# --------------------------------------------------------------------------


def test_F_0_has_a_single_trailing_component():
    """L=0 的角度部分恆為 1，所以最後一軸是 2L+1 = 1。"""
    f0 = layers.F_0(RBF_COUNT, output_dim=3)
    assert f0(rbf_expansion(geometry(5))).shape == (5, 5, 3, 1)


def test_F_0_is_rotation_invariant():
    f0 = layers.F_0(RBF_COUNT, output_dim=3)
    points = geometry(5)
    rotation = utils.random_rotation_matrix(7)
    before = f0(rbf_expansion(points))
    after = f0(rbf_expansion(points @ rotation.T))
    assert torch.allclose(after, before, atol=1e-5)


def test_F_0_is_translation_invariant():
    f0 = layers.F_0(RBF_COUNT, output_dim=3)
    points = geometry(5)
    shift = torch.tensor([1.5, -2.0, 0.7])
    before = f0(rbf_expansion(points))
    after = f0(rbf_expansion(points + shift))
    assert torch.allclose(after, before, atol=1e-5)


# --------------------------------------------------------------------------
# F_1：L = 1 濾波器
# --------------------------------------------------------------------------


def test_F_1_has_three_trailing_components():
    f1 = layers.F_1(RBF_COUNT, output_dim=3)
    points = geometry(5)
    out = f1(rbf_expansion(points), utils.difference_matrix(points))
    assert out.shape == (5, 5, 3, 3)


def test_F_1_angular_part_is_a_unit_vector():
    """非對角元素的角度部分範數為 1；輸出的長度全由徑向部分決定。"""
    points = geometry(5)
    rij = utils.difference_matrix(points)
    off_diagonal = ~torch.eye(5, dtype=torch.bool)
    norms = torch.linalg.vector_norm(layers.unit_vectors(rij), dim=-1)[off_diagonal]
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_F_1_points_along_the_pair_separation():
    """輸出方向就是 rij 的方向（徑向權重為負時反向），不能有側向分量。"""
    f1 = layers.F_1(RBF_COUNT, output_dim=3)
    points = geometry(5)
    rij = utils.difference_matrix(points)
    out = f1(rbf_expansion(points), rij)
    # 平行 <=> 外積為 0
    cross = torch.linalg.cross(out, rij.unsqueeze(-2).expand_as(out), dim=-1)
    assert torch.allclose(cross, torch.zeros_like(cross), atol=1e-5)


def test_F_1_is_zero_on_the_diagonal():
    """i = j 時 rij = 0，沒有方向可言，輸出必須是 0 而不是 NaN。"""
    f1 = layers.F_1(RBF_COUNT, output_dim=3)
    points = geometry(5)
    out = f1(rbf_expansion(points), utils.difference_matrix(points))
    diagonal = out[range(5), range(5)]
    assert not torch.isnan(out).any()
    assert torch.equal(diagonal, torch.zeros_like(diagonal))


def test_F_1_has_no_nan_gradient_at_coincident_points():
    f1 = layers.F_1(RBF_COUNT, output_dim=3)
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    points.requires_grad_(True)
    f1(rbf_expansion(points), utils.difference_matrix(points)).sum().backward()
    assert points.grad is not None
    assert not torch.isnan(points.grad).any()


def test_F_1_mask_removes_the_gradient_at_coincident_points():
    """遮罩擋掉的是梯度，不是數值。

    unit_vectors 在 rij = 0 已經回 0（分母被 EPSILON 夾住），所以就算不遮罩，
    重合點的輸出也是 0。差別在反向：那裡的 Jacobian 是 1/sqrt(EPSILON) = 1e4
    乘上徑向權重，會把一個純屬正則化假象的巨大梯度灌回座標。遮罩之後這一項
    恰好是 0。

    對角線驗不出這件事——rij[i, i] 不論座標怎麼動都是 0，梯度本來就是 0。
    要兩個**相異**但重合的點才問得出來。
    """
    f1 = layers.F_1(RBF_COUNT, output_dim=3)
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    points.requires_grad_(True)
    out = f1(rbf_expansion(points), utils.difference_matrix(points))
    grad = torch.autograd.grad(out[0, 1].sum(), points)[0]
    assert torch.equal(grad, torch.zeros_like(grad))


def test_F_1_is_rotation_equivariant():
    """座標轉 R，輸出的最後一軸就跟著轉 R。這是整條鏈的關卡。"""
    f1 = layers.F_1(RBF_COUNT, output_dim=3)
    points = geometry(5)
    rotation = utils.random_rotation_matrix(7)
    before = f1(rbf_expansion(points), utils.difference_matrix(points))
    rotated = points @ rotation.T
    after = f1(rbf_expansion(rotated), utils.difference_matrix(rotated))
    assert torch.allclose(after, before @ rotation.T, atol=1e-5)


def test_F_1_is_not_accidentally_rotation_invariant():
    """反面對照：F_1 的輸出真的會被旋轉改變。

    沒有這一條的話，一支恆回 0 的實作也能通過上面的等變性測試。
    """
    f1 = layers.F_1(RBF_COUNT, output_dim=3)
    points = geometry(5)
    rotation = utils.random_rotation_matrix(7)
    before = f1(rbf_expansion(points), utils.difference_matrix(points))
    rotated = points @ rotation.T
    after = f1(rbf_expansion(rotated), utils.difference_matrix(rotated))
    assert not torch.allclose(after, before, atol=1e-3)


def test_F_1_is_translation_invariant():
    f1 = layers.F_1(RBF_COUNT, output_dim=3)
    points = geometry(5)
    shift = torch.tensor([1.5, -2.0, 0.7])
    before = f1(rbf_expansion(points), utils.difference_matrix(points))
    after = f1(rbf_expansion(points + shift), utils.difference_matrix(points + shift))
    assert torch.allclose(after, before, atol=1e-5)


# --------------------------------------------------------------------------
# CG 收縮：三條濾波路徑
# --------------------------------------------------------------------------

CHANNELS = 2
POINTS = 5

# 三條路徑，以及各自吃什麼 L、吐什麼 L。filter_0 與 filter_1_output_1
# 各接受兩種輸入 L，所以是五個組合。
PATHS = [
    ("filter_0 : L=0 -> L=0", layers.filter_0, 0, 0),
    ("filter_0 : L=1 -> L=1", layers.filter_0, 1, 1),
    ("filter_1_output_0 : L=1 -> L=0", layers.filter_1_output_0, 1, 0),
    ("filter_1_output_1 : L=0 -> L=1", layers.filter_1_output_1, 0, 1),
    ("filter_1_output_1 : L=1 -> L=1", layers.filter_1_output_1, 1, 1),
]
PATH_IDS = [path[0] for path in PATHS]


def features(angular_momentum: int, channels: int = CHANNELS, seed: int = 1) -> torch.Tensor:
    """``[N, C, 2L+1]`` 的輸入特徵。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(POINTS, channels, 2 * angular_momentum + 1, generator=g)


def apply_path(path: torch.nn.Module, layer_input: torch.Tensor, points: torch.Tensor):
    """呼叫方式差在要不要 rij——filter_0 的濾波器不帶角度部分。"""
    rbf = rbf_expansion(points)
    if isinstance(path, layers.filter_0):
        return path(layer_input, rbf)
    return path(layer_input, rbf, utils.difference_matrix(points))


def rotate(t: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """把最後一軸長度 3 的張量整批旋轉。"""
    return t @ rotation.T


# --- 形狀與契約 -----------------------------------------------------------


@pytest.mark.parametrize("angular_momentum", [0, 1])
def test_filter_0_preserves_the_input_angular_momentum(angular_momentum):
    """L × 0 → L：濾波器不帶角動量，輸入的 L 原封不動傳出去。"""
    path = layers.filter_0(RBF_COUNT, output_dim=CHANNELS)
    x = features(angular_momentum)
    out = apply_path(path, x, geometry(POINTS))
    assert out.shape == (POINTS, CHANNELS, 2 * angular_momentum + 1)


def test_filter_1_output_0_yields_a_scalar():
    path = layers.filter_1_output_0(RBF_COUNT, output_dim=CHANNELS)
    out = apply_path(path, features(1), geometry(POINTS))
    assert out.shape == (POINTS, CHANNELS, 1)


def test_filter_1_output_0_rejects_scalar_input():
    """0 × 1 只能耦合出 L=1，產不出 L=0——這條路徑不存在。"""
    path = layers.filter_1_output_0(RBF_COUNT, output_dim=CHANNELS)
    with pytest.raises(ValueError, match="0 x 1 cannot yield 0"):
        apply_path(path, features(0), geometry(POINTS))


@pytest.mark.parametrize("angular_momentum", [0, 1])
def test_filter_1_output_1_yields_a_vector(angular_momentum):
    path = layers.filter_1_output_1(RBF_COUNT, output_dim=CHANNELS)
    out = apply_path(path, features(angular_momentum), geometry(POINTS))
    assert out.shape == (POINTS, CHANNELS, 3)


@pytest.mark.parametrize("cls", [layers.filter_1_output_0, layers.filter_1_output_1])
def test_L_1_filters_reject_unsupported_angular_momenta(cls):
    """L=2 以上還沒移植，要明講而不是算出一個形狀對、意思錯的東西。

    filter_0 不在此列：它的 CG 是 eye(2L+1)，對任何 L 都成立，上游同樣
    沒有這個分支。會卡住的是帶角動量的濾波器——那才需要真正的 CG 表。
    """
    path = cls(RBF_COUNT, output_dim=CHANNELS)
    with pytest.raises(NotImplementedError):
        apply_path(path, features(2), geometry(POINTS))


def test_filter_0_works_for_any_angular_momentum():
    """反過來說，L × 0 → L 對 L=2 也是對的，不該擋。"""
    path = layers.filter_0(RBF_COUNT, output_dim=CHANNELS)
    out = apply_path(path, features(2), geometry(POINTS))
    assert out.shape == (POINTS, CHANNELS, 5)


# --- 符號釘死 -------------------------------------------------------------
#
# 本票最大的風險：einsum 索引排錯、或 ε 符號弄反，訓練 loss 照樣會降
# （訓練集只有單一朝向，背起來就好），要到測試集才發現。所以這裡把每條
# 路徑的收縮結果與手算的張量運算逐元素比對，不看 loss。
#
# 手算時直接取用 path.filter 的輸出：要釘的是「收縮」這一步，不是濾波器
# 本身——濾波器已經在上面驗過了。


def test_filter_0_contraction_is_a_scalar_rescale():
    """CG 是單位矩陣，所以收縮就是「用徑向權重加權、對鄰居求和」。"""
    path = layers.filter_0(RBF_COUNT, output_dim=CHANNELS)
    points = geometry(POINTS)
    x = features(1)
    rbf = rbf_expansion(points)

    out = path(x, rbf)
    f0 = path.filter(rbf)  # [N, N, C, 1]
    manual = (f0 * x.unsqueeze(0)).sum(dim=1)  # 對鄰居 b 求和

    assert torch.allclose(out, manual, atol=1e-6)


def test_filter_1_output_0_contraction_is_an_inner_product():
    """1 × 1 → 0 的 CG 是 δ_jk，收縮就是內積。"""
    path = layers.filter_1_output_0(RBF_COUNT, output_dim=CHANNELS)
    points = geometry(POINTS)
    x = features(1)
    rbf, rij = rbf_expansion(points), utils.difference_matrix(points)

    out = path(x, rbf, rij)
    f1 = path.filter(rbf, rij)  # [N, N, C, 3]
    manual = (f1 * x.unsqueeze(0)).sum(dim=-1, keepdim=True).sum(dim=1)

    assert torch.allclose(out, manual, atol=1e-6)


def test_filter_1_output_1_contraction_on_scalars_is_a_rescale():
    """0 × 1 → 1 的 CG 是 eye(3)：純量只縮放濾波器的方向，不改方向。"""
    path = layers.filter_1_output_1(RBF_COUNT, output_dim=CHANNELS)
    points = geometry(POINTS)
    x = features(0)
    rbf, rij = rbf_expansion(points), utils.difference_matrix(points)

    out = path(x, rbf, rij)
    f1 = path.filter(rbf, rij)
    manual = (f1 * x.unsqueeze(0)).sum(dim=1)  # x 的最後一軸長度 1，會廣播

    assert torch.allclose(out, manual, atol=1e-6)


def test_filter_1_output_1_contraction_is_exactly_the_cross_product():
    """1 × 1 → 1 的 CG 是 Levi-Civita，收縮逐元素等於外積——含符號。

    ε 的符號弄反、或 einsum 把 j / k 排反，結果就是外積取負，這條會抓到。
    順序是 filter × input：ε_ijk F_j x_k = (F × x)_i。
    """
    path = layers.filter_1_output_1(RBF_COUNT, output_dim=CHANNELS)
    points = geometry(POINTS)
    x = features(1)
    rbf, rij = rbf_expansion(points), utils.difference_matrix(points)

    out = path(x, rbf, rij)
    f1 = path.filter(rbf, rij)  # [N, N, C, 3]
    manual = torch.linalg.cross(f1, x.unsqueeze(0).expand_as(f1), dim=-1).sum(dim=1)

    assert torch.allclose(out, manual, atol=1e-6)


def test_filter_1_output_1_is_not_the_negated_cross_product():
    """反面對照：確認上一條不是在比對兩個都反了號的東西。"""
    path = layers.filter_1_output_1(RBF_COUNT, output_dim=CHANNELS)
    points = geometry(POINTS)
    x = features(1)
    rbf, rij = rbf_expansion(points), utils.difference_matrix(points)

    out = path(x, rbf, rij)
    f1 = path.filter(rbf, rij)
    flipped = torch.linalg.cross(x.unsqueeze(0).expand_as(f1), f1, dim=-1).sum(dim=1)

    assert not torch.allclose(out, flipped, atol=1e-3)


# --- 等變性（本票關卡）----------------------------------------------------


@pytest.mark.parametrize(("name", "cls", "l_in", "l_out"), PATHS, ids=PATH_IDS)
def test_path_is_rotation_equivariant(name, cls, l_in, l_out):
    """座標與輸入特徵一起轉 R，輸出就照它自己的 L 跟著轉。

    L=0 的輸出不動、L=1 的輸出轉 R。三條路徑各自獨立驗，不靠組合結果掩護。
    """
    path = cls(RBF_COUNT, output_dim=CHANNELS)
    points = geometry(POINTS)
    x = features(l_in)
    rotation = utils.random_rotation_matrix(7)

    before = apply_path(path, x, points)
    rotated_x = rotate(x, rotation) if l_in == 1 else x
    after = apply_path(path, rotated_x, points @ rotation.T)

    expected = rotate(before, rotation) if l_out == 1 else before
    assert torch.allclose(after, expected, atol=1e-5)


@pytest.mark.parametrize(("name", "cls", "l_in", "l_out"), PATHS, ids=PATH_IDS)
def test_path_is_translation_invariant(name, cls, l_in, l_out):
    """濾波器只看點對之間的相對位置，整體平移看不見。"""
    path = cls(RBF_COUNT, output_dim=CHANNELS)
    points = geometry(POINTS)
    x = features(l_in)
    shift = torch.tensor([1.5, -2.0, 0.7])

    before = apply_path(path, x, points)
    after = apply_path(path, x, points + shift)

    assert torch.allclose(after, before, atol=1e-5)


# --------------------------------------------------------------------------
# SelfInteraction：沿通道軸的線性混合
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angular_momentum", [0, 1])
def test_self_interaction_mixes_channels_and_keeps_L(angular_momentum):
    """每個點各自做，不看鄰居；動的是通道數，不是 L。"""
    si = layers.SelfInteraction(CHANNELS, 7, bias=False)
    out = si(features(angular_momentum))
    assert out.shape == (POINTS, 7, 2 * angular_momentum + 1)


def test_self_interaction_weights_are_orthogonal():
    """對齊作者的 orthogonal_initializer。

    輸出通道少於輸入時，正交的是列（W Wᵀ = I）。
    """
    si = layers.SelfInteraction(8, 4, bias=False)
    w = si.linear.weight
    assert torch.allclose(w @ w.T, torch.eye(4), atol=1e-5)


def test_self_interaction_bias_starts_at_zero():
    si = layers.SelfInteraction(CHANNELS, CHANNELS, bias=True)
    assert si.linear.bias is not None
    assert torch.equal(si.linear.bias, torch.zeros_like(si.linear.bias))


def test_self_interaction_without_bias_has_no_bias_parameter():
    si = layers.SelfInteraction(CHANNELS, CHANNELS, bias=False)
    assert si.linear.bias is None
    assert len(list(si.parameters())) == 1


def test_self_interaction_without_bias_is_rotation_equivariant():
    si = layers.SelfInteraction(CHANNELS, 3, bias=False)
    x = features(1)
    rotation = utils.random_rotation_matrix(7)
    assert torch.allclose(si(rotate(x, rotation)), rotate(si(x), rotation), atol=1e-5)


def test_self_interaction_with_bias_breaks_equivariance_for_L_1():
    """這就是 L>0 不能加 bias 的理由，直接驗給它看。

    bias 是每個通道一個純量，加在 2L+1 個分量上——那是個固定的向量，
    不會跟著旋轉，所以它一進來等變性就破了。bias 初始化是 0，要填非零
    才問得出這件事。
    """
    si = layers.SelfInteraction(CHANNELS, 3, bias=True)
    with torch.no_grad():
        si.linear.bias.fill_(0.5)
    x = features(1)
    rotation = utils.random_rotation_matrix(7)
    assert not torch.allclose(si(rotate(x, rotation)), rotate(si(x), rotation), atol=1e-3)


def test_self_interaction_with_bias_is_safe_for_L_0():
    """L=0 那邊 bias 無害，所以純量才用有 bias 版。

    直接對 L=0 特徵轉一轉是問不出東西的——它本來就不隨旋轉變。要讓旋轉
    真的進到式子裡，得從座標出發：filter_1_output_0 把 L=1 特徵縮成 L=0，
    再過有 bias 的 SelfInteraction，整條仍該是旋轉不變的。
    """
    path = layers.filter_1_output_0(RBF_COUNT, output_dim=CHANNELS)
    si = layers.SelfInteraction(CHANNELS, 3, bias=True)
    with torch.no_grad():
        si.linear.bias.fill_(0.5)
    points = geometry(POINTS)
    x = features(1)
    rotation = utils.random_rotation_matrix(7)

    before = si(apply_path(path, x, points))
    after = si(apply_path(path, rotate(x, rotation), points @ rotation.T))

    assert torch.allclose(after, before, atol=1e-5)


def test_self_interaction_is_translation_invariant_through_a_filter():
    """SelfInteraction 自己看不到座標，平移不變要接在濾波器後面才問得出來。"""
    path = layers.filter_1_output_1(RBF_COUNT, output_dim=CHANNELS)
    si = layers.SelfInteraction(CHANNELS, 3, bias=False)
    points = geometry(POINTS)
    x = features(1)
    shift = torch.tensor([1.5, -2.0, 0.7])

    before = si(apply_path(path, x, points))
    after = si(apply_path(path, x, points + shift))

    assert torch.allclose(after, before, atol=1e-5)


# --------------------------------------------------------------------------
# Nonlinearity：只動範數、不動方向
# --------------------------------------------------------------------------


def test_nonlinearity_on_scalars_is_the_bare_activation():
    """L=0 直接套用非線性，不加 bias——與作者一致。"""
    nl = layers.Nonlinearity(CHANNELS, angular_momentum=0)
    x = features(0)
    assert torch.allclose(nl(x), F.elu(x), atol=1e-6)


def test_nonlinearity_on_scalars_has_no_parameters():
    """L=0 那條沒有 bias，所以整個模組不該掛任何參數。

    上游在這裡仍然建了一個 biases 變數卻沒用到；照抄會多出一顆死參數，
    optimizer 還是會替它配狀態。
    """
    nl = layers.Nonlinearity(CHANNELS, angular_momentum=0)
    assert list(nl.parameters()) == []


def test_nonlinearity_defaults_to_elu_and_can_switch_to_ssp():
    x = features(0)
    assert torch.allclose(layers.Nonlinearity(CHANNELS, 0)(x), F.elu(x), atol=1e-6)
    assert torch.allclose(
        layers.Nonlinearity(CHANNELS, 0, nonlin=utils.ssp)(x), utils.ssp(x), atol=1e-6
    )


def test_nonlinearity_on_vectors_has_one_bias_per_channel():
    """L>0 的 bias 加在範數上——那是個旋轉不變量，所以不破壞等變性。"""
    nl = layers.Nonlinearity(CHANNELS, angular_momentum=1)
    assert nl.bias is not None
    assert nl.bias.shape == (CHANNELS,)
    assert torch.equal(nl.bias, torch.zeros(CHANNELS))


def test_nonlinearity_on_vectors_keeps_the_direction():
    """只縮放範數、不改方向——方向一動就破壞等變性。"""
    nl = layers.Nonlinearity(CHANNELS, angular_momentum=1)
    x = features(1)
    cosine = F.cosine_similarity(nl(x), x, dim=-1)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-5)


def test_nonlinearity_on_vectors_actually_changes_the_norm():
    """反面對照：確認上一條不是因為它根本沒動輸入而通過。"""
    nl = layers.Nonlinearity(CHANNELS, angular_momentum=1, nonlin=utils.ssp)
    x = features(1)
    assert not torch.allclose(nl(x), x, atol=1e-3)


def test_nonlinearity_rejects_input_of_the_wrong_L():
    """建構時就綁定 L（bias 的形狀取決於它），餵錯要當場講。"""
    nl = layers.Nonlinearity(CHANNELS, angular_momentum=1)
    with pytest.raises(ValueError, match="angular_momentum"):
        nl(features(0))


def test_nonlinearity_on_vectors_is_rotation_equivariant():
    """輸出與輸入平行、縮放倍率只看範數（旋轉不變量），所以整支跟著轉。"""
    nl = layers.Nonlinearity(CHANNELS, angular_momentum=1)
    x = features(1)
    rotation = utils.random_rotation_matrix(7)
    assert torch.allclose(nl(rotate(x, rotation)), rotate(nl(x), rotation), atol=1e-5)


def test_nonlinearity_on_scalars_is_rotation_invariant():
    """同 SelfInteraction 的 L=0 那條：要接在濾波器後面，旋轉才進得了式子。"""
    path = layers.filter_1_output_0(RBF_COUNT, output_dim=CHANNELS)
    nl = layers.Nonlinearity(CHANNELS, angular_momentum=0)
    points = geometry(POINTS)
    x = features(1)
    rotation = utils.random_rotation_matrix(7)

    before = nl(apply_path(path, x, points))
    after = nl(apply_path(path, rotate(x, rotation), points @ rotation.T))

    assert torch.allclose(after, before, atol=1e-5)


def test_nonlinearity_is_translation_invariant_through_a_filter():
    path = layers.filter_1_output_1(RBF_COUNT, output_dim=CHANNELS)
    nl = layers.Nonlinearity(CHANNELS, angular_momentum=1)
    points = geometry(POINTS)
    x = features(1)
    shift = torch.tensor([1.5, -2.0, 0.7])

    before = nl(apply_path(path, x, points))
    after = nl(apply_path(path, x, points + shift))

    assert torch.allclose(after, before, atol=1e-5)


# --------------------------------------------------------------------------
# Convolution 與 concatenation：完整一層
# --------------------------------------------------------------------------


def scalar_input(channels: int = 1) -> dict[int, list[torch.Tensor]]:
    """網路的起點：每個點一組全 1 的 L=0 特徵（同上游 notebook 的 embed 前身）。"""
    return {0: [torch.ones(POINTS, channels, 1)]}


def count_parameters_by_hand(layer: layers.Layer) -> int:
    """繞過 nn.Module 的註冊機制，照結構把子模組的參數逐一數一遍。

    子模組若被塞進普通 list / dict 而不是 ModuleList / ModuleDict，
    layer.parameters() 就看不到它們，但這裡照樣數得到——兩邊對不起來
    就是註冊漏了。
    """
    leaves = list(layer.convolution.paths)
    for step in (layer.self_interaction, layer.nonlinearity):
        for module_list in step.layers.values():
            # ModuleDict.values() 的靜態型別只到 Module，實際上是 ModuleList
            leaves.extend(cast(torch.nn.ModuleList, module_list))
    return sum(len(list(leaf.parameters())) for leaf in leaves)


# --- 通道帳 ---------------------------------------------------------------


def test_convolution_channel_bookkeeping_from_a_single_scalar():
    """{0:[1]} 進去，出來是 {0:[1], 1:[1]}——L=0 走 filter_0 與 filter_1_output_1。"""
    conv = layers.Convolution({0: [1]}, RBF_COUNT)
    assert conv.output_channels == {0: [1], 1: [1]}


def test_convolution_channel_bookkeeping_from_mixed_L():
    """{0:[4], 1:[4]} 進去，出來是 {0:[4,4], 1:[4,4,4]}——五條路徑。

    順序照 reference 的走訪：先 L=0（filter_0 → L=0、filter_1_output_1 → L=1），
    再 L=1（filter_0 → L=1、filter_1_output_0 → L=0、filter_1_output_1 → L=1）。
    """
    conv = layers.Convolution({0: [4], 1: [4]}, RBF_COUNT)
    assert conv.output_channels == {0: [4, 4], 1: [4, 4, 4]}


def test_concatenated_channel_bookkeeping():
    """沿通道軸串接，一個 L 只剩一條張量。"""
    assert layers.concatenated_channels({0: [4, 4], 1: [4, 4, 4]}) == {0: [8], 1: [12]}


def test_convolution_rejects_angular_momenta_it_cannot_build():
    conv_input = {0: [4], 2: [4]}
    with pytest.raises(NotImplementedError):
        layers.Convolution(conv_input, RBF_COUNT)


# --- 中間狀態的型別與可分步呼叫 -------------------------------------------


def test_convolution_output_matches_its_own_bookkeeping():
    """通道帳是建構時算的，要真的等於 forward 出來的形狀。"""
    conv = layers.Convolution({0: [4], 1: [4]}, RBF_COUNT)
    points = geometry(POINTS)
    x = {0: [features(0, channels=4)], 1: [features(1, channels=4)]}

    out = conv(x, rbf_expansion(points), utils.difference_matrix(points))

    for angular_momentum, channel_counts in conv.output_channels.items():
        assert [t.shape[-2] for t in out[angular_momentum]] == channel_counts
        for t in out[angular_momentum]:
            assert t.shape == (POINTS, t.shape[-2], 2 * angular_momentum + 1)


def test_intermediate_state_is_a_dict_of_lists_of_tensors():
    layer = layers.Layer({0: [1]}, RBF_COUNT, output_dim=4)
    points = geometry(POINTS)
    rbf, rij = rbf_expansion(points), utils.difference_matrix(points)

    after_convolution = layer.convolution(scalar_input(), rbf, rij)
    after_concatenation = layers.concatenation(after_convolution)

    for state in (after_convolution, after_concatenation):
        assert isinstance(state, dict)
        assert all(isinstance(key, int) for key in state)
        assert all(isinstance(value, list) for value in state.values())
        assert all(isinstance(t, torch.Tensor) for value in state.values() for t in value)


def test_the_four_steps_compose_into_the_whole_layer():
    """四個步驟各自可單獨呼叫，串起來就等於 Layer。"""
    layer = layers.Layer({0: [1]}, RBF_COUNT, output_dim=4)
    points = geometry(POINTS)
    rbf, rij = rbf_expansion(points), utils.difference_matrix(points)
    x = scalar_input()

    stepwise = layer.nonlinearity(
        layer.self_interaction(layers.concatenation(layer.convolution(x, rbf, rij)))
    )
    whole = layer(x, rbf, rij)

    assert stepwise.keys() == whole.keys()
    for angular_momentum in whole:
        for got, expected in zip(stepwise[angular_momentum], whole[angular_momentum], strict=True):
            assert torch.equal(got, expected)


def test_concatenation_leaves_one_tensor_per_L():
    conv = layers.Convolution({0: [4], 1: [4]}, RBF_COUNT)
    points = geometry(POINTS)
    x = {0: [features(0, channels=4)], 1: [features(1, channels=4)]}

    out = layers.concatenation(conv(x, rbf_expansion(points), utils.difference_matrix(points)))

    assert [t.shape[-2] for t in out[0]] == [8]
    assert [t.shape[-2] for t in out[1]] == [12]


def test_self_interaction_step_uses_bias_only_for_scalars():
    """L=0 有 bias、L>0 沒有——票 05 的規則要在這一層真的被套用。"""
    step = layers.SelfInteractionStep({0: [8], 1: [12]}, output_dim=4)
    # 有 bias 的版本掛 weight + bias 兩顆參數，無 bias 的只有 weight
    assert len(list(step.layers["0"].parameters())) == 2
    assert len(list(step.layers["1"].parameters())) == 1


# --- 參數註冊 -------------------------------------------------------------


def test_every_submodule_is_registered():
    """子模組放進普通 list 或 dict，optimizer 就看不到它們。

    訓練照跑、loss 也會降，但那些層永遠不學——這是 nn.Module 重構最典型的坑。
    """
    layer = layers.Layer({0: [1]}, RBF_COUNT, output_dim=4)
    assert len(list(layer.parameters())) == count_parameters_by_hand(layer)


def test_parameter_count_matches_an_explicit_tally():
    """獨立於上一條的絕對數字，防止兩邊一起漏或一起重複數。

    {0:[1]} 起手、rbf_count=4、output_dim=4：
      Convolution 兩條路徑（filter_0、filter_1_output_1），各一個 R
        = 2 條 × (w1, b1, w2, b2) = 8
      SelfInteraction L=0 有 bias（w, b）+ L=1 無 bias（w）    = 3
      Nonlinearity   L=0 無參數 + L=1 一根 bias                = 1
    """
    layer = layers.Layer({0: [1]}, RBF_COUNT, output_dim=4)
    assert len(list(layer.parameters())) == 8 + 3 + 1


# --- 完整一層的等變性 -----------------------------------------------------


def test_layer_is_rotation_equivariant():
    """L=0 的輸出在旋轉下不變、L=1 的輸出跟著轉。

    輸入是全 1 的 L=0 特徵，本身不隨旋轉變；所以這裡量到的完全是這一層
    自己的等變性。
    """
    layer = layers.Layer({0: [1]}, RBF_COUNT, output_dim=4)
    points = geometry(POINTS)
    rotation = utils.random_rotation_matrix(7)

    def run(pts):
        return layer(scalar_input(), rbf_expansion(pts), utils.difference_matrix(pts))

    before = run(points)
    after = run(points @ rotation.T)

    assert torch.allclose(after[0][0], before[0][0], atol=1e-5)
    assert torch.allclose(after[1][0], rotate(before[1][0], rotation), atol=1e-5)


def test_layer_output_is_not_trivially_constant():
    """反面對照：L=1 的輸出真的會被旋轉改變，等變性不是靠全零蒙混。"""
    layer = layers.Layer({0: [1]}, RBF_COUNT, output_dim=4)
    points = geometry(POINTS)
    rotation = utils.random_rotation_matrix(7)

    def run(pts):
        return layer(scalar_input(), rbf_expansion(pts), utils.difference_matrix(pts))

    assert not torch.allclose(run(points @ rotation.T)[1][0], run(points)[1][0], atol=1e-3)


def test_layer_is_translation_invariant():
    layer = layers.Layer({0: [1]}, RBF_COUNT, output_dim=4)
    points = geometry(POINTS)
    shift = torch.tensor([1.5, -2.0, 0.7])

    def run(pts):
        return layer(scalar_input(), rbf_expansion(pts), utils.difference_matrix(pts))

    before = run(points)
    after = run(points + shift)

    for angular_momentum in before:
        assert torch.allclose(after[angular_momentum][0], before[angular_momentum][0], atol=1e-5)


def test_layers_stack():
    """一層的輸出通道帳就是下一層的輸入通道帳——票 07 靠這個疊三層。"""
    first = layers.Layer({0: [1]}, RBF_COUNT, output_dim=4)
    second = layers.Layer(first.output_channels, RBF_COUNT, output_dim=4)
    points = geometry(POINTS)
    rbf, rij = rbf_expansion(points), utils.difference_matrix(points)
    rotation = utils.random_rotation_matrix(11)

    def run(pts):
        r, d = rbf_expansion(pts), utils.difference_matrix(pts)
        return second(first(scalar_input(), r, d), r, d)

    assert first.output_channels == {0: [4], 1: [4]}
    out = second(first(scalar_input(), rbf, rij), rbf, rij)
    assert out[0][0].shape == (POINTS, 4, 1)
    assert out[1][0].shape == (POINTS, 4, 3)

    before, after = run(points), run(points @ rotation.T)
    assert torch.allclose(after[0][0], before[0][0], atol=1e-5)
    assert torch.allclose(after[1][0], rotate(before[1][0], rotation), atol=1e-5)
