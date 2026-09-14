"""實驗二之二：轉動慣量張量（論文 §5.2）。

移植對照：reference/tensorfieldnetworks-tf/moment_of_inertia.ipynb

給一堆隨機點和它們的質量，指定其中一個當中心，問**繞著那個中心轉有多難**。
難的程度不是一個數字——繞不同軸的難度不同——所以答案是一個 3×3 對稱矩陣。

網路型別是 ``0 → 0 ⊕ 2``。一個 3×3 對稱矩陣有 6 個獨立數字，而這 6 個在旋轉
之下分成兩堆互不相干的：1 個管整體大小（trace，L=0），5 個管形狀（無 trace
的對稱部分，L=2）。網路必須分別產生這兩堆，最後再用一個固定的換算式拼回矩陣
——中間不能有可學的全連接層，因為任意權重會把「旋轉時不動的」和「會轉的」加
在一起，加完就哪條規則都不遵守了。

結構與重力那半一樣極簡：單層、單通道、沒有 self-interaction、沒有非線性。
差別只在這裡有兩條並排的路徑，所以有**兩條**徑向函數可學。

驗收也一樣是曲線而不是準確率：訓練完把兩條徑向函數畫出來，它們應該分別長成
``2/3·r²`` 與 ``−r²``。
"""

import numpy as np
import torch
from torch import Tensor, nn

from tfn.layers import filter_0, filter_2_output_2, matrix_from_0_2, probe_radial
from tfn.utils import (
    FLOAT_TYPE,
    difference_matrix,
    distance_matrix,
    l2_loss,
    rbf_expansion,
    ssp,
)

__all__ = [
    "NUM_POINTS",
    "MAX_COORD",
    "MASS_LOW",
    "MASS_HIGH",
    "CENTRE_INDEX",
    "RBF_LOW",
    "RBF_HIGH",
    "RBF_COUNT",
    "TRAIN_STEPS",
    "LEARNING_RATE",
    "VALIDATION_SAMPLES",
    "FIT_LOW",
    "FIT_HIGH",
    "VALIDATION_LOSS_CEILING",
    "RADIAL_L0_NRMSE_CEILING",
    "RADIAL_L2_NRMSE_CEILING",
    "random_points_and_masses",
    "moment_of_inertia",
    "analytic_radial_0",
    "analytic_radial_2",
    "MomentOfInertiaModel",
    "step_loss",
    "train",
    "validation_loss",
]

# 資料產生器，同上游 notebook
NUM_POINTS = 15
MAX_COORD = 0.5
MASS_LOW = 0.5
MASS_HIGH = 2.0
CENTRE_INDEX = 0

# RBF 設定，同上游 notebook（與重力那半相同）
RBF_LOW = 0.0
RBF_HIGH = 2.0
RBF_COUNT = 30

TRAIN_STEPS = 10001
LEARNING_RATE = 1e-4
VALIDATION_SAMPLES = 1000

# 曲線的比對區間：實測 20000 組樣本後「中心點到其他點」距離的 p5–p95，
# 涵蓋 90% 的資料。
#
# 上游 notebook 的圖把上限標在 sqrt(3·(rbf_high/4)²) = 0.866，那是**點到原點**
# 的最大距離。但相關的是「中心點到其他點」的距離，而中心點自己也是隨機的——
# 實測最大值是 1.60，有 22% 的資料落在 0.866 之外。那條線標錯了。
FIT_LOW = 0.26
FIT_HIGH = 1.07

# 門檻由實測決定，抓法與重力那半相同——看的是「訓練後」與「沒學起來」之間的
# 距離，不是訓練的隨機波動：
#
#   指標              訓練後    換訓練 seed      未訓練     門檻
#   validation loss   0.0120   0.0011 – 0.0151   254.3      0.10
#   nRMSE（L=0）      0.0282   0.0078 – 0.0282     1.87      0.06
#   nRMSE（L=2）      0.0110   0.0094 – 0.0110     1.69      0.06
#
# validation loss 換個 seed 會差 14 倍，所以只負責擋「根本沒學起來」——0.10
# 離未訓練還有 2500 倍。nRMSE 才是關卡，取實測最差值的兩倍；符號整條反轉是
# 2.0、未訓練接近 2，兩個失敗模式都離 0.06 很遠。
#
# 兩條曲線共用同一個門檻值：L=2 實測其實更緊（0.0110 vs 0.0282），但只觀察過
# 三個 seed，沒有足夠證據把它訂得更嚴。
VALIDATION_LOSS_CEILING = 0.10
RADIAL_L0_NRMSE_CEILING = 0.06
RADIAL_L2_NRMSE_CEILING = 0.06


def random_points_and_masses(
    rng: int | np.random.Generator | None = None,
    *,
    num_points: int = NUM_POINTS,
    max_coord: float = MAX_COORD,
    mass_low: float = MASS_LOW,
    mass_high: float = MASS_HIGH,
    centre_index: int = CENTRE_INDEX,
    device: torch.device | str | None = None,
    dtype: torch.dtype = FLOAT_TYPE,
) -> tuple[Tensor, Tensor]:
    """一組隨機點雲與質量。``-> ([num_points, 3], [num_points])``

    點數固定（不像重力那半會浮動），每軸均勻分布於 ``±max_coord``。

    **中心點的質量設成 0**，這一步不是可有可無的：L=0 濾波器沒有「自己對自己」
    的遮罩，所以中心點自己的質量會經由 ``R₀(距離≈0)`` 直接漏進輸出。解析解那邊
    它本來就貢獻 0（相對位置是零向量），把輸入質量歸零是讓網路那一側對齊。

    收斂之後其實不需要這一步——真解 ``R₀(r) = 2/3·r²`` 在 r = 0 就是 0——但
    訓練早期它會給網路一個「拿自我項當免費 bias」的誘因，把曲線在近距離帶歪。
    """
    generator = np.random.default_rng(rng)
    points = generator.uniform(-max_coord, max_coord, size=(num_points, 3))
    masses = generator.uniform(mass_low, mass_high, size=num_points)
    masses[centre_index] = 0.0
    return (
        torch.as_tensor(points, device=device, dtype=dtype),
        torch.as_tensor(masses, device=device, dtype=dtype),
    )


def moment_of_inertia(points: Tensor, masses: Tensor, index: int = CENTRE_INDEX) -> Tensor:
    """繞 ``points[index]`` 的轉動慣量張量。``([N, 3], [N]) -> [3, 3]``

    ``I_ij = Σ_p m_p (|r_p|² δ_ij − (r_p)_i (r_p)_j)``，位置相對於中心點。

    這是 ground truth，不是網路的一部分。中心點自己的 ``r`` 是零向量，所以不論
    質量多少都貢獻 0。
    """
    relative = points - points[index]
    squared = torch.sum(relative * relative, dim=-1)
    outer = relative.unsqueeze(-1) * relative.unsqueeze(-2)
    identity = torch.eye(3, device=points.device, dtype=points.dtype)
    per_point = squared.reshape(-1, 1, 1) * identity - outer
    return (masses.reshape(-1, 1, 1) * per_point).sum(dim=0)


def analytic_radial_0(distances: Tensor) -> Tensor:
    """L=0 那條徑向函數的解析解：``R₀(r) = 2/3·r²``。``[M] -> [M]``

    來自把轉動慣量拆成 trace 與非 trace 兩半。``trace(I) = Σ m (3r² − r²)``
    ``= 2 Σ m r²``，所以各向同性那一份是 ``(2/3) Σ m r² · I``。而 L=0 路徑
    算的是 ``Σ_b R₀(d) · m_b``，兩式對起來就只剩這一個解。
    """
    return 2.0 / 3.0 * distances**2


def analytic_radial_2(distances: Tensor) -> Tensor:
    """L=2 那條徑向函數的解析解：``R₂(r) = −r²``。``[M] -> [M]``

    扣掉 trace 之後剩下的是 ``−Σ m r² (r̂r̂ᵀ − I/3)``。而 L=2 路徑經過
    :func:`~tfn.layers.matrix_from_0_2` 之後算的正是
    ``Σ_b R₂(d) · m_b · (r̂r̂ᵀ − I/3)``（票 10 的封閉形式），所以 ``R₂(r) = −r²``。

    網路裡沒有別的自由度能吸收這些係數，所以兩條曲線的尺度都是絕對釘死的，
    可以直接比數值，不需要先做尺度擬合。
    """
    return -(distances**2)


class MomentOfInertiaModel(nn.Module):
    """論文 §5.2 的轉動慣量網路。``([N, 3], [N]) -> [N, 3, 3]``

    兩條並排的卷積路徑，各一個通道：``L × 0 → L`` 產生純量那一份，
    ``0 × 2 → 2`` 產生五個 L=2 分量。兩者用固定的換算式拼成矩陣。

    ``matrix_from_0_2`` 不是一層網路，裡面沒有參數——中間放一個可學的全連接層
    會直接破壞等變性。整張網路可學的只有兩條徑向函數。

    輸出是**每個點一個矩陣**（卷積天生如此），但只有中心點那個有意義；
    :func:`step_loss` 只取那一個。

    通道數固定是 1（論文如此），所以 :meth:`forward` 可以安全地 squeeze 掉
    通道軸。
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
        # 徑向 MLP 的非線性是 shifted softplus（同作者，不是 R 預設的 relu）。
        # bias 走預設的全零——與重力那半不同，那份 notebook 明確傳了 glorot。
        self.path_0 = filter_0(rbf_count, output_dim=1, nonlin=ssp)
        self.path_2 = filter_2_output_2(rbf_count, output_dim=1, nonlin=ssp)

    def forward(self, points: Tensor, masses: Tensor) -> Tensor:
        rij = difference_matrix(points)
        rbf = rbf_expansion(
            distance_matrix(points), low=self.rbf_low, high=self.rbf_high, count=self.rbf_count
        )
        # [N] -> [N, 1, 1]：一條通道，L=0 所以最後一軸長度 1
        features = masses.reshape(-1, 1, 1)

        # [N, 1, 1] -> [N]，[N, 1, 5] -> [N, 5]
        scalar = self.path_0(features, rbf).squeeze(-1).squeeze(-1)
        l2 = self.path_2(features, rbf, rij).squeeze(-2)
        return matrix_from_0_2(scalar, l2)

    def radial_curves(self, distances: Tensor) -> tuple[Tensor, Tensor]:
        """兩條學到的徑向函數。``[M] -> ([M], [M])``，依序是 L=0 與 L=2。

        驗收就是拿這兩條分別跟 :func:`analytic_radial_0`、:func:`analytic_radial_2`
        比。走 ``probe_radial`` 而不是自己展開 RBF，讓畫圖與訓練用同一組設定。
        """
        return (
            probe_radial(
                self.path_0.filter.radial,
                distances,
                low=self.rbf_low,
                high=self.rbf_high,
                count=self.rbf_count,
            )[:, 0],
            probe_radial(
                self.path_2.filter.radial,
                distances,
                low=self.rbf_low,
                high=self.rbf_high,
                count=self.rbf_count,
            )[:, 0],
        )


def step_loss(
    model: MomentOfInertiaModel, points: Tensor, masses: Tensor, index: int = CENTRE_INDEX
) -> Tensor:
    """一個樣本的 loss：差的平方和除以 2，**只取中心點**那一個 3×3。

    跟重力不同——轉動慣量是「繞著某個中心」才有定義的，所以得先挑一個中心，
    其餘 14 個點算出來的矩陣丟掉。
    """
    predicted = model(points, masses)[index]
    return l2_loss(predicted - moment_of_inertia(points, masses, index))


def train(
    model: MomentOfInertiaModel,
    *,
    steps: int = TRAIN_STEPS,
    learning_rate: float = LEARNING_RATE,
    rng: int | np.random.Generator | None = None,
    device: torch.device | str | None = None,
) -> list[float]:
    """訓練，回傳每一步的 loss。

    沒有資料集、沒有 epoch：每一步現場亂生一組新的點雲，答案用物理公式直接算。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(rng)
    history = []
    for _ in range(steps):
        points, masses = random_points_and_masses(generator, device=device)
        optimizer.zero_grad()
        loss = step_loss(model, points, masses)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history


@torch.no_grad()
def validation_loss(
    model: MomentOfInertiaModel,
    *,
    samples: int = VALIDATION_SAMPLES,
    rng: int | np.random.Generator | None = None,
    device: torch.device | str | None = None,
) -> float:
    """在一批新生的點雲上量平均 loss。

    每個樣本都是固定 15 個點、固定比 9 個數字，所以量級一致——不像重力那邊
    點數浮動會讓每步的 loss 量級跟著跳。
    """
    generator = np.random.default_rng(rng)
    total = 0.0
    for _ in range(samples):
        points, masses = random_points_and_masses(generator, device=device)
        total += step_loss(model, points, masses).item()
    return total / samples
