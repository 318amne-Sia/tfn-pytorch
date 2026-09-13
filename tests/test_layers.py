"""tfn.layers 的濾波器測試：徑向部分 R 與角度部分 F_0 / F_1。

移植對照：reference/tensorfieldnetworks-tf/tensorfieldnetworks/layers.py

本檔的關卡是等變性：F_0 的輸出在旋轉下不變、F_1 的輸出跟著轉。
兩者都靠票 02 的 random_rotation_matrix，那支已在 test_utils 驗過是真旋轉。
"""

import math

import pytest
import torch

from tfn import layers, utils

# 與上游 notebook 第 3 格相同的 RBF 設定。這段屬於實驗設定（票 07 的
# notebook）而不是濾波器本身，所以留在測試裡當 fixture，不進 tfn。
RBF_LOW = 0.0
RBF_HIGH = 3.5
RBF_COUNT = 4


def rbf_expansion(geometry: torch.Tensor) -> torch.Tensor:
    """``[N, 3] -> [N, N, RBF_COUNT]``，高斯基底展開的點對距離。

    距離是旋轉不變也是平移不變的，所以 rbf 在等變性測試裡是常數——
    F_1 輸出會轉，靠的全是角度部分。
    """
    spacing = (RBF_HIGH - RBF_LOW) / RBF_COUNT
    centers = torch.linspace(RBF_LOW, RBF_HIGH, RBF_COUNT)
    gamma = 1.0 / spacing
    dij = utils.distance_matrix(geometry)
    return torch.exp(-gamma * (dij.unsqueeze(-1) - centers) ** 2)


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
