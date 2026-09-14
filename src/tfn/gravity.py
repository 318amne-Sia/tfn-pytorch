"""實驗二之一：牛頓重力（論文 §5.2）。

移植對照：reference/tensorfieldnetworks-tf/gravity.ipynb

餵進去一堆隨機點和它們的質量，網路要吐出每個點受到的重力加速度向量。
網路型別是 ``0 → 1``：輸入是純量（質量），輸出是向量（加速度）。

結構上跟實驗一是兩回事。這裡只有**一層、一條路徑、一個通道**，沒有
self-interaction、沒有非線性、最後也沒有全連接層——因為輸出本身就是會
跟著座標旋轉的東西，不能退回普通神經網路。整張網路可學的只有一條徑向
函數，所以訓練完可以把它整條畫出來，跟課本的 ``−1/r²`` 逐點對照。

這也是為什麼這個實驗的驗收不是準確率而是一條曲線：論文要展示的不是
「網路學得會」，而是「學到的東西攤開來看，跟物理公式對得上」。
"""

import numpy as np
import torch
from torch import Tensor, nn

from tfn.layers import filter_1_output_1, probe_radial
from tfn.utils import (
    EPSILON,
    FLOAT_TYPE,
    difference_matrix,
    distance_matrix,
    l2_loss,
    rbf_expansion,
    ssp,
)

__all__ = [
    "MAX_COORD",
    "MIN_SEPARATION",
    "MASS_LOW",
    "MASS_HIGH",
    "TRAIN_MAX_POINTS",
    "VALIDATION_MAX_POINTS",
    "RBF_LOW",
    "RBF_HIGH",
    "RBF_COUNT",
    "TRAIN_STEPS",
    "LEARNING_RATE",
    "VALIDATION_SAMPLES",
    "VALIDATION_LOSS_CEILING",
    "RADIAL_NRMSE_CEILING",
    "random_points_and_masses",
    "accelerations",
    "analytic_radial",
    "GravityModel",
    "step_loss",
    "train",
    "validation_loss",
]

# 資料產生器，同上游 notebook
MAX_COORD = 2.0
MIN_SEPARATION = 0.5
MASS_LOW = 0.5
MASS_HIGH = 2.0
TRAIN_MAX_POINTS = 10
# 驗證刻意用更多點：網路學的是「一對點之間的關係」，跟總共有幾個點無關，
# 所以換一個點數還成立是一個不花錢的泛化檢查
VALIDATION_MAX_POINTS = 50

# RBF 設定，同上游 notebook。注意中心只鋪到 2.0，但實測 73% 的點對距離
# 超過 2.0、56% 超過 2.5——更遠的距離在 RBF 上全是 0，網路一律吐同一個
# 常數。那不算災難（−1/r² 在那裡本來就趨近 0），但代表學到的曲線只在
# [MIN_SEPARATION, RBF_HIGH] 這一段有意義。
RBF_LOW = 0.0
RBF_HIGH = 2.0
RBF_COUNT = 30

TRAIN_STEPS = 1001
LEARNING_RATE = 1e-3
VALIDATION_SAMPLES = 1000

# 兩個門檻都是實測出來的，不是猜的。門檻要擋的是「移植錯了」這種等級的
# 偏差，不是訓練的隨機波動，所以餘裕是照兩者的距離抓的：
#
#   指標              訓練後    換訓練 seed    未訓練      門檻
#   validation loss    1.88      1.6 – 4.5     2832       10.0
#   nRMSE              0.099     0.076–0.099   0.655       0.20
#
# validation loss 對「換一個訓練 seed」很敏感（2.4 倍），但對「換一批驗證
# 資料」不敏感（±10%）——而且那個 loss 最大的模型曲線反而是最準的。它量的
# 是收斂程度不是曲線品質，所以門檻放得鬆，只負責擋住「根本沒學起來」；
# 未訓練是 2832，10.0 仍然有 283 倍的距離。
#
# nRMSE 才是本票真正的關卡，門檻取實測最差值的兩倍。整條曲線符號反轉會是
# 2.0，未訓練是 0.655，兩個失敗模式都離 0.20 很遠。
VALIDATION_LOSS_CEILING = 10.0
RADIAL_NRMSE_CEILING = 0.20


def random_points_and_masses(
    rng: int | np.random.Generator | None = None,
    *,
    max_points: int = TRAIN_MAX_POINTS,
    max_coord: float = MAX_COORD,
    min_separation: float = MIN_SEPARATION,
    mass_low: float = MASS_LOW,
    mass_high: float = MASS_HIGH,
    device: torch.device | str | None = None,
    dtype: torch.dtype = FLOAT_TYPE,
) -> tuple[Tensor, Tensor]:
    """一組隨機點雲與質量。``-> ([N, 3], [N])``

    先抽 2 到 ``max_points`` 個候選點，再**依序**把與任何一個已保留點距離
    小於 ``min_separation`` 的剔掉，所以回傳的 N 是浮動的，而且可能低到 1。

    剔除近距離的點不是可有可無：目標是 ``1/r²``，兩點靠太近時正確答案會
    爆炸到很大，訓練會被那幾個極端樣本帶走。

    實測約 0.04% 的呼叫會只剩 1 個點——那一步沒有任何梯度。這是預期行為，
    訓練迴圈要撐得住，不要當成 bug 去追。

    ``rng`` 可以是 seed、numpy ``Generator``、或 None。傳 Generator 會推進
    它的狀態，所以訓練迴圈裡連續呼叫得到不同的點雲。

    上游的點座標走 python ``random``、質量走 ``np.random``，兩套亂數源；
    這裡統一用一個 Generator。反正 TF1 與 PyTorch 的權重初始化本來就不可能
    逐位元對齊，保留兩套亂數源沒有好處。
    """
    generator = np.random.default_rng(rng)
    # 同上游的 random.randint，兩端都含
    count = int(generator.integers(2, max_points + 1))
    candidates = generator.uniform(-max_coord, max_coord, size=(count, 3))

    kept: list[np.ndarray] = []
    for candidate in candidates:
        if all(np.linalg.norm(candidate - previous) >= min_separation for previous in kept):
            kept.append(candidate)

    masses = generator.uniform(mass_low, mass_high, size=len(kept))
    return (
        torch.as_tensor(np.asarray(kept), device=device, dtype=dtype),
        torch.as_tensor(masses, device=device, dtype=dtype),
    )


def accelerations(points: Tensor, masses: Tensor) -> Tensor:
    """牛頓重力下每個點的加速度。``([N, 3], [N]) -> [N, 3]``

    ``a_i = −Σ_j m_j · r̂_ij / d_ij²``，其中 ``r_ij := r_i − r_j``，取 ``G = 1``。

    這是 ground truth，不是網路的一部分。它自己必須是等變的——否則就是在
    拿一組會破壞等變性的標準答案去教一個等變的網路。

    自我項明確遮掉，而不是靠 ``d³ + EPSILON`` 把它壓成 0。後者碰巧也會得到
    0（分子的 ``r_ij`` 同樣是 0），但那是巧合不是意圖，上游那邊寫的也是一個
    明確的「兩點相同就跳過」。
    """
    count = points.shape[0]
    rij = difference_matrix(points)
    # 未正則化的 norm：這裡不需要 distance_matrix 那層保護，對角線本來就要是 0
    dij = torch.linalg.vector_norm(rij, dim=-1)
    contribution = -rij * masses.reshape(1, -1, 1) / (dij**3 + EPSILON).unsqueeze(-1)
    self_pairs = torch.eye(count, dtype=torch.bool, device=points.device).unsqueeze(-1)
    return torch.where(self_pairs, torch.zeros_like(contribution), contribution).sum(dim=1)


def analytic_radial(distances: Tensor) -> Tensor:
    """徑向函數的解析解：``R(r) = −1/r²``。``[M] -> [M]``

    推導只有一行。濾波器算的是 ``Σ_j R(d_ij) · m_j · r̂_ij``，而正確答案是
    ``Σ_j −m_j · r̂_ij / d_ij²``——兩式逐項對起來，徑向函數唯一的解就是
    ``−1/r²``。網路裡沒有別的自由度可以吸收這個係數，所以曲線的尺度是被
    釘死的，可以直接比數值，不需要先做尺度擬合。
    """
    return -1.0 / distances**2


class GravityModel(nn.Module):
    """論文 §5.2 的重力網路。``([N, 3], [N]) -> [N, 3]``

    一個 L=1 卷積路徑，一個通道，就這樣。質量是 L=0 特徵餵進去，加速度是
    L=1 特徵掉出來；``0 ⊗ 1 → 1`` 的 CG 是單位矩陣，所以收縮那一步只是讓
    純量去縮放濾波器的方向。

    通道數固定是 1（論文如此），所以 :meth:`forward` 可以安全地把通道軸
    squeeze 掉。要改通道數的話那一行要一起改。
    """

    def __init__(
        self,
        rbf_count: int = RBF_COUNT,
        rbf_low: float = RBF_LOW,
        rbf_high: float = RBF_HIGH,
    ) -> None:
        super().__init__()
        self.rbf_count = rbf_count
        self.rbf_low = rbf_low
        self.rbf_high = rbf_high
        # 兩個非預設值都是上游 notebook 明確傳進去的：徑向 MLP 的非線性是
        # shifted softplus（不是 R 預設的 relu），bias 走 glorot（不是全零）。
        # 後者在這裡有感——b2 的界是 ±1.732，而這個實驗只跑 1001 步。
        self.path = filter_1_output_1(rbf_count, output_dim=1, nonlin=ssp, bias_init="glorot")

    def forward(self, points: Tensor, masses: Tensor) -> Tensor:
        rij = difference_matrix(points)
        rbf = rbf_expansion(
            distance_matrix(points), low=self.rbf_low, high=self.rbf_high, count=self.rbf_count
        )
        # [N] -> [N, 1, 1]：一條通道，L=0 所以最後一軸長度 1
        features = masses.reshape(-1, 1, 1)
        # [N, 1, 3] -> [N, 3]
        return self.path(features, rbf, rij).squeeze(-2)

    def radial_curve(self, distances: Tensor) -> Tensor:
        """學到的徑向函數在 ``distances`` 上的值。``[M] -> [M]``

        驗收就是拿這條跟 :func:`analytic_radial` 比。走 ``probe_radial`` 而
        不是自己展開 RBF，是為了讓畫圖與訓練用的是同一組 RBF 設定。
        """
        curve = probe_radial(
            self.path.filter.radial,
            distances,
            low=self.rbf_low,
            high=self.rbf_high,
            count=self.rbf_count,
        )
        return curve[:, 0]


def step_loss(model: GravityModel, points: Tensor, masses: Tensor) -> Tensor:
    """一個樣本的 loss：差的平方和除以 2，**涵蓋全部的點**。

    跟轉動慣量不同——加速度在每個點上同時都有定義，不必先挑一個中心。
    """
    return l2_loss(model(points, masses) - accelerations(points, masses))


def train(
    model: GravityModel,
    *,
    steps: int = TRAIN_STEPS,
    learning_rate: float = LEARNING_RATE,
    rng: int | np.random.Generator | None = None,
    max_points: int = TRAIN_MAX_POINTS,
    device: torch.device | str | None = None,
) -> list[float]:
    """訓練，回傳每一步的 loss。

    沒有資料集、沒有 epoch：每一步都現場亂生一組新的點雲，答案用物理公式
    直接算。資料無限多，而且整個網路只有一條曲線可學，沒有過擬合的空間。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(rng)
    history = []
    for _ in range(steps):
        points, masses = random_points_and_masses(generator, max_points=max_points, device=device)
        optimizer.zero_grad()
        loss = step_loss(model, points, masses)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history


@torch.no_grad()
def validation_loss(
    model: GravityModel,
    *,
    samples: int = VALIDATION_SAMPLES,
    rng: int | np.random.Generator | None = None,
    max_points: int = VALIDATION_MAX_POINTS,
    device: torch.device | str | None = None,
) -> float:
    """在一批新生的點雲上量平均 loss。

    每個樣本的 loss 是**加總**而不是平均，所以點數多的樣本本來就貢獻比較
    大——這裡跟著上游不做正規化，只把樣本之間取平均。
    """
    generator = np.random.default_rng(rng)
    total = 0.0
    for _ in range(samples):
        points, masses = random_points_and_masses(generator, max_points=max_points, device=device)
        total += step_loss(model, points, masses).item()
    return total / samples
