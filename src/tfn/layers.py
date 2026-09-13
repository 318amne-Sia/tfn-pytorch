"""濾波器：徑向部分 R 與角度部分 F_0 / F_1。

移植對照：reference/tensorfieldnetworks-tf/tensorfieldnetworks/layers.py

論文 §4.1：濾波器寫成 ``F(r) = R(‖r‖) · Y(r̂)``——一個只看距離的徑向函數，
乘上一個只看方向的球諧函數。等變性全部住在角度部分：R 吃的是距離，旋轉下
不動；Y 是 L 階球諧，旋轉下按 L 階表示變換。

上游還有 F_2（L=2），實驗一用不到，先不移植。

型別上這裡與上游有一個結構差異：TF1 靠 variable_scope 在呼叫時憑空生參數，
PyTorch 的參數屬於物件，所以 R / F_0 / F_1 都是 nn.Module，建構時就要知道
輸入維度。
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tfn.utils import EPSILON, norm_with_epsilon

__all__ = ["R", "F_0", "F_1", "unit_vectors"]


def unit_vectors(v: Tensor, dim: int = -1) -> Tensor:
    """沿 ``dim`` 正規化成單位向量。

    分母走 :func:`~tfn.utils.norm_with_epsilon`，所以零向量回傳的是
    ``0 / 1e-4 = 0`` 而不是 NaN。
    """
    return v / norm_with_epsilon(v, dim=dim, keepdim=True)


class R(nn.Module):
    """徑向函數：兩層 MLP，把 RBF 展開映成每個點對、每個通道的徑向權重。

    ``[N, N, input_dim] -> [N, N, output_dim]``

    論文 §5 說這一段與 SchNet 相同：距離先展開成高斯基底（呼叫端負責，見
    票 07 的 notebook），再過 MLP。這裡不做展開，只吃展開後的結果。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int | None = None,
        nonlin: Callable[[Tensor], Tensor] = F.relu,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            # 同作者：未指定就取輸入維度
            hidden_dim = input_dim
        self.nonlin = nonlin
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        for linear in (self.linear1, self.linear2):
            # 明確對齊作者的 xavier_initializer / constant_initializer(0.)。
            # nn.Linear 預設是 Kaiming uniform（界窄 sqrt(3) 倍）加上均勻亂數
            # bias，沿用預設等於偷偷換掉了初始化。
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, rbf: Tensor) -> Tensor:
        return self.linear2(self.nonlin(self.linear1(rbf)))


class F_0(nn.Module):
    """L = 0 濾波器。``[N, N, input_dim] -> [N, N, output_dim, 1]``

    L=0 的角度部分是 0 階球諧，也就是常數 1，所以這支就是 :class:`R` 再補一根
    長度 2L+1 = 1 的軸，好跟 L>0 的濾波器共用同一組收縮寫法（票 04）。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int | None = None,
        nonlin: Callable[[Tensor], Tensor] = F.relu,
    ) -> None:
        super().__init__()
        self.radial = R(input_dim, output_dim=output_dim, hidden_dim=hidden_dim, nonlin=nonlin)

    def forward(self, rbf: Tensor) -> Tensor:
        return self.radial(rbf).unsqueeze(-1)


class F_1(nn.Module):
    """L = 1 濾波器。``([N, N, input_dim], [N, N, 3]) -> [N, N, output_dim, 3]``

    角度部分是 1 階球諧，在實座標下就是點對之間的單位向量 ``r̂ij``——旋轉座標，
    它就跟著旋轉，濾波器的等變性完全來自這裡。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int | None = None,
        nonlin: Callable[[Tensor], Tensor] = F.relu,
    ) -> None:
        super().__init__()
        self.radial = R(input_dim, output_dim=output_dim, hidden_dim=hidden_dim, nonlin=nonlin)

    def forward(self, rbf: Tensor, rij: Tensor) -> Tensor:
        # [N, N, output_dim]
        radial = self.radial(rbf)

        # rij = 0 的點對（i = j，或兩個點恰好重合）沒有方向可言，遮掉。
        #
        # 這裡刻意用未正則化的 vector_norm，不是 utils.distance_matrix：後者走
        # norm_with_epsilon，對角線是 1e-4，拿去比 `< EPSILON`（1e-8）永遠為假，
        # 遮罩會靜靜地失效。上游此處用的也是未正則化的 tf.norm。
        #
        # 遮罩擋掉的其實是梯度而不是數值：unit_vectors 在 rij = 0 已經回 0，
        # 沒有遮罩輸出也是 0；但反向時那裡的 Jacobian 是 1/sqrt(EPSILON) = 1e4，
        # 會把一個純屬正則化假象的巨大梯度灌回座標。
        dij = torch.linalg.vector_norm(rij, dim=-1)
        masked_radial = torch.where((dij < EPSILON).unsqueeze(-1), 0.0, radial)

        # [N, N, 1, 3] * [N, N, output_dim, 1] -> [N, N, output_dim, 3]
        return unit_vectors(rij).unsqueeze(-2) * masked_radial.unsqueeze(-1)
