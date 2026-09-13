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

from tfn.utils import EPSILON, get_eijk, norm_with_epsilon

__all__ = [
    "R",
    "F_0",
    "F_1",
    "unit_vectors",
    "filter_0",
    "filter_1_output_0",
    "filter_1_output_1",
]


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


# --------------------------------------------------------------------------
# CG 收縮：三條濾波路徑
# --------------------------------------------------------------------------
#
# 論文 §4.2：把 L_f 階的濾波器與 L_in 階的輸入特徵，依 Clebsch-Gordan 係數
# 耦合成 L_out 階的輸出。角動量耦合規則 |L_f - L_in| ≤ L_out ≤ L_f + L_in
# 決定了哪些路徑存在——實驗一只用到 L ≤ 1，於是剩下四種組合、三個模組。
#
# 這裡的 CG 都沒有做歸一化（真正的係數還差一個常數倍）。那個常數是每條路徑
# 一個全域純量，會被 R 的權重吸收掉，所以照上游省略。


def _contract(cg: Tensor, filter_out: Tensor, layer_input: Tensor) -> Tensor:
    """CG 收縮。三條路徑共用這一個式子，差別只在 ``cg``。

    ``([2Lo+1, 2Lf+1, 2Li+1], [N, N, C, 2Lf+1], [N, C, 2Li+1]) -> [N, C, 2Lo+1]``

    索引::

        a  輸出的那個點
        b  鄰居；它被求和掉——這一步就是「卷積」的加總
        f  通道。濾波器與輸入共用同一根通道軸，所以濾波器的 output_dim
           必須等於輸入的通道數（呼叫端負責，見票 06 的通道帳）
        i  輸出的角動量分量、j 濾波器的、k 輸入的；三者由 cg 綁在一起
    """
    return torch.einsum("ijk,abfj,bfk->afi", cg, filter_out, layer_input)


# 三個路徑類別沿用上游的小寫命名（同 F_0 / F_1），方便與 reference 逐行對照。
class filter_0(nn.Module):
    """L × 0 → L。``([N, C, 2L+1], [N, N, input_dim]) -> [N, C, 2L+1]``

    濾波器不帶角動量（L_f = 0），耦合規則只允許 L_out = L_in，CG 是單位矩陣。
    直觀上就是：用徑向權重把鄰居的特徵加權求和，方向資訊原封不動。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int | None = None,
        nonlin: Callable[[Tensor], Tensor] = F.relu,
    ) -> None:
        super().__init__()
        self.filter = F_0(input_dim, output_dim=output_dim, hidden_dim=hidden_dim, nonlin=nonlin)

    def forward(self, layer_input: Tensor, rbf: Tensor) -> Tensor:
        # [N, N, C, 1]
        filter_out = self.filter(rbf)
        dim = layer_input.shape[-1]
        # [2L+1, 1, 2L+1]：中間那根是濾波器的角度軸，L_f = 0 所以長度 1
        cg = torch.eye(dim, device=layer_input.device, dtype=layer_input.dtype).unsqueeze(-2)
        return _contract(cg, filter_out, layer_input)


class filter_1_output_0(nn.Module):
    """1 × 1 → 0。``([N, C, 3], [N, N, input_dim], [N, N, 3]) -> [N, C, 1]``

    兩個向量縮成純量，CG 是 δ_jk，收縮就是內積。這是網路把方向資訊轉回
    旋轉不變量的唯一管道——票 07 最後拿去分類的就是這種 L=0 特徵。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int | None = None,
        nonlin: Callable[[Tensor], Tensor] = F.relu,
    ) -> None:
        super().__init__()
        self.filter = F_1(input_dim, output_dim=output_dim, hidden_dim=hidden_dim, nonlin=nonlin)

    def forward(self, layer_input: Tensor, rbf: Tensor, rij: Tensor) -> Tensor:
        # [N, N, C, 3]
        filter_out = self.filter(rbf, rij)
        dim = layer_input.shape[-1]
        if dim == 1:
            # 0 ⊗ 1 只耦合得出 L=1，這條路徑不存在。要 L=1 輸出請用
            # filter_1_output_1。
            raise ValueError("0 x 1 cannot yield 0：L=0 的輸入配 L=1 的濾波器只能得到 L=1")
        if dim != 3:
            raise NotImplementedError(f"L > 1 尚未移植（輸入的最後一軸為 {dim}）")
        # [1, 3, 3]
        cg = torch.eye(3, device=layer_input.device, dtype=layer_input.dtype).unsqueeze(0)
        return _contract(cg, filter_out, layer_input)


class filter_1_output_1(nn.Module):
    """L × 1 → 1。``([N, C, 2L+1], [N, N, input_dim], [N, N, 3]) -> [N, C, 3]``

    輸入 L=0 時 CG 是 ``eye(3)``：純量只縮放濾波器的方向。
    輸入 L=1 時 CG 是 Levi-Civita，收縮起來恰好是外積 ``F × x``。

    外積這條是本票的符號風險所在：ε 的符號或 einsum 的 j / k 排反，結果會
    整個取負，而 loss 照樣降得下去。測試拿 ``torch.linalg.cross`` 逐元素釘死。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int | None = None,
        nonlin: Callable[[Tensor], Tensor] = F.relu,
    ) -> None:
        super().__init__()
        self.filter = F_1(input_dim, output_dim=output_dim, hidden_dim=hidden_dim, nonlin=nonlin)

    def forward(self, layer_input: Tensor, rbf: Tensor, rij: Tensor) -> Tensor:
        # [N, N, C, 3]
        filter_out = self.filter(rbf, rij)
        dim = layer_input.shape[-1]
        if dim == 1:
            # 0 × 1 -> 1，CG 形狀 [3, 3, 1]
            cg = torch.eye(3, device=layer_input.device, dtype=layer_input.dtype).unsqueeze(-1)
        elif dim == 3:
            # 1 × 1 -> 1，CG 形狀 [3, 3, 3]
            cg = get_eijk(device=layer_input.device, dtype=layer_input.dtype)
        else:
            raise NotImplementedError(f"L > 1 尚未移植（輸入的最後一軸為 {dim}）")
        return _contract(cg, filter_out, layer_input)
