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
from typing import NamedTuple, cast

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
    "SelfInteraction",
    "Nonlinearity",
    "Convolution",
    "concatenation",
    "concatenated_channels",
    "SelfInteractionStep",
    "NonlinearityStep",
    "Layer",
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

    論文 §5 說這一段與 SchNet 相同：距離先展開成高斯基底
    （:func:`~tfn.utils.rbf_expansion`，由呼叫端負責），再過 MLP。
    這裡不做展開，只吃展開後的結果。
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


# --------------------------------------------------------------------------
# 一層裡不涉及點對關係的兩個運算
# --------------------------------------------------------------------------


class SelfInteraction(nn.Module):
    """沿通道軸的線性混合。``[N, C_in, 2L+1] -> [N, C_out, 2L+1]``

    每個點各自做，不看鄰居；動的是通道數，L 原封不動。角動量那根軸完全沒被
    碰到，所以（不加 bias 的話）等變性是白拿的。

    ``bias`` 沒有預設值，必須明講：**L=0 用 True、L>0 用 False**。bias 是每個
    通道一個純量，會被加到 2L+1 個分量上——那等於加了一個固定向量，它不隨
    座標旋轉，L>0 的等變性一加就破。這件事錯了不會報錯也不會讓 loss 變難看，
    所以寧可讓呼叫端每次都寫出來。

    實作上就是一個 nn.Linear 作用在通道軸上：它的 weight 形狀 ``[C_out, C_in]``
    恰好等於上游 ``w_si`` 的形狀，bias 形狀 ``[C_out]`` 也對得上。上游那句
    einsum 加轉置（``'afi,gf->aig'`` 再 permute）算的是同一件事。
    """

    def __init__(self, input_dim: int, output_dim: int, *, bias: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        # 對齊作者的 orthogonal_initializer / constant_initializer(0.)
        nn.init.orthogonal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        # 通道軸換到最後讓 nn.Linear 吃，算完換回來
        return self.linear(x.transpose(-2, -1)).transpose(-2, -1)


class Nonlinearity(nn.Module):
    """旋轉等變的非線性。``[N, C, 2L+1] -> [N, C, 2L+1]``

    L>0 時只縮放特徵的範數、不改變方向：非線性作用在範數上（那是旋轉不變量），
    得到的倍率再乘回原向量。方向一動就破壞等變性，所以不能逐元素套用。

    L=0 時直接套用非線性，**不加 bias**——與作者一致。論文 §4.3 寫的是
    ``η(‖V‖ + b)``，L=0 時退化成 ``η(V + b)``；但這一步的前面永遠是
    self-interaction，而 L=0 的 self-interaction 已經加過一個 per-channel
    bias 了，再加一個只是把它重新參數化成 ``b₁ + b₂``，多一組永遠學不出
    獨立作用的參數。

    L 在建構時就綁定，因為 L>0 的 bias 形狀取決於它——不像票 04 的濾波路徑
    可以等到 forward 再看輸入的形狀決定。上游在 L=0 時仍然建了一個 biases
    變數卻沒用到，這裡不照抄那顆死參數。
    """

    def __init__(
        self,
        channels: int,
        angular_momentum: int,
        nonlin: Callable[[Tensor], Tensor] = F.elu,
    ) -> None:
        super().__init__()
        self.angular_momentum = angular_momentum
        self.nonlin = nonlin
        if angular_momentum == 0:
            # 同 nn.Linear(bias=False) 的作法：明確登記成 None，而不是
            # 擺一個普通屬性
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        expected = 2 * self.angular_momentum + 1
        if x.shape[-1] != expected:
            raise ValueError(
                f"angular_momentum={self.angular_momentum} 需要最後一軸為 {expected}，"
                f"收到 {x.shape[-1]}"
            )
        if self.bias is None:
            return self.nonlin(x)

        # [N, C]：範數是旋轉不變量，bias 與非線性都安全地作用在這上面
        norm = norm_with_epsilon(x, dim=-1)
        factor = self.nonlin(norm + self.bias) / norm
        return x * factor.unsqueeze(-1)


# --------------------------------------------------------------------------
# 論文 §5.1 的一個 module：convolution → concatenation → self-interaction
#                          → nonlinearity
# --------------------------------------------------------------------------
#
# 中間狀態一路都是 Features，也就是 dict[L, list[Tensor]]：鍵是角動量，值是
# 「這個 L 目前累積了哪幾條特徵」。convolution 會讓 list 變長（每條輸入分裂成
# 好幾條路徑的輸出），concatenation 再把它壓回每個 L 一條。
#
# 這四個步驟各自是獨立可呼叫的單位，與論文的四個具名步驟一比一對應。
# 帶參數的三個是 nn.Module，concatenation 沒有參數所以是純函式。
#
# 名字上的取捨：SelfInteractionStep / NonlinearityStep 的 Step 後綴只是為了
# 跟票 05 那兩個「作用在單一張量上」的同名模組區分——它們是這裡的零件。
# Convolution 沒有這個困擾，所以不加後綴。

Features = dict[int, list[Tensor]]
Channels = dict[int, list[int]]

SUPPORTED_L = (0, 1)


class _Path(NamedTuple):
    """一條卷積路徑要吃哪條輸入、結果算誰的。

    ``paths[k]`` 與 ``_plan[k]`` 是平行的兩份：模組本體必須住在 nn.ModuleList
    裡才會被註冊，這些純資料則不能塞進去。
    """

    l_in: int
    index: int
    l_out: int
    needs_rij: bool


class Convolution(nn.Module):
    """把每條輸入特徵沿所有合法路徑跑一遍，依輸出的 L 收集。

    ``(Features, [N, N, rbf_count], [N, N, 3]) -> Features``

    路徑在**建構時**就依 ``input_channels``（``{L: [每條特徵的通道數]}``）決定
    好，而不是等 forward 看張量形狀——PyTorch 不像 TF1 能延遲建參數，所以下一層
    要知道自己會收到什麼，得先問得到 :attr:`output_channels`。

    走訪順序照 reference：先依 L 由小到大，同一個 L 內依序處理每條特徵，每條
    特徵依序試 ``L×0→L``、``1×1→0``（只有 L=1 有）、``L×1→1``。順序會決定
    concatenation 之後通道的排列，所以照抄。
    """

    def __init__(
        self,
        input_channels: Channels,
        rbf_count: int,
        hidden_dim: int | None = None,
        nonlin: Callable[[Tensor], Tensor] = F.relu,
    ) -> None:
        super().__init__()
        unsupported = sorted(set(input_channels) - set(SUPPORTED_L))
        if unsupported:
            raise NotImplementedError(f"L > 1 尚未移植（收到 L={unsupported}）")

        paths: list[nn.Module] = []
        plan: list[_Path] = []
        self.output_channels: Channels = {0: [], 1: []}

        def add(
            path: nn.Module, l_in: int, index: int, channels: int, l_out: int, needs_rij: bool
        ) -> None:
            paths.append(path)
            plan.append(_Path(l_in, index, l_out, needs_rij))
            self.output_channels[l_out].append(channels)

        for l_in in sorted(input_channels):
            for index, channels in enumerate(input_channels[l_in]):
                # 濾波器與輸入共用同一根通道軸（見 _contract），所以濾波器的
                # output_dim 必須等於這條輸入的通道數
                kwargs = {"output_dim": channels, "hidden_dim": hidden_dim, "nonlin": nonlin}
                # L × 0 → L：濾波器不帶角動量，輸入的 L 原樣傳出
                add(filter_0(rbf_count, **kwargs), l_in, index, channels, l_in, needs_rij=False)
                if l_in == 1:
                    # 1 × 1 → 0：內積，把方向資訊收回成純量
                    add(
                        filter_1_output_0(rbf_count, **kwargs),
                        l_in,
                        index,
                        channels,
                        0,
                        needs_rij=True,
                    )
                # L × 1 → 1
                add(
                    filter_1_output_1(rbf_count, **kwargs),
                    l_in,
                    index,
                    channels,
                    1,
                    needs_rij=True,
                )

        self.paths = nn.ModuleList(paths)
        self._plan = plan

    def forward(self, features: Features, rbf: Tensor, rij: Tensor) -> Features:
        output: Features = {0: [], 1: []}
        for path, plan in zip(self.paths, self._plan, strict=True):
            x = features[plan.l_in][plan.index]
            output[plan.l_out].append(path(x, rbf, rij) if plan.needs_rij else path(x, rbf))
        return output


def concatenation(features: Features) -> Features:
    """沿通道軸串接，每個 L 壓回一條特徵。

    卷積把一條輸入分裂成好幾條路徑的輸出，這一步把同一個 L 的重新併成一條，
    後面的 self-interaction 才有單一的通道軸可以混。
    """
    return {angular_momentum: [torch.cat(ts, dim=-2)] for angular_momentum, ts in features.items()}


def concatenated_channels(channels: Channels) -> Channels:
    """:func:`concatenation` 對應的通道帳。"""
    return {angular_momentum: [sum(cs)] for angular_momentum, cs in channels.items()}


class _PerTensorStep(nn.Module):
    """對 Features 裡每一條張量各作用一個子模組的步驟。

    子模組住在 ``nn.ModuleDict[str, nn.ModuleList]``：ModuleDict 的鍵只能是
    字串，所以 L 要轉成 str。用普通的 dict / list 會讓這些子模組完全不出現在
    ``parameters()`` 裡——訓練照跑、loss 也會降，但它們永遠不學。
    """

    def __init__(self, modules_by_l: dict[int, list[nn.Module]]) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(angular_momentum): nn.ModuleList(ms)
                for angular_momentum, ms in modules_by_l.items()
            }
        )

    def forward(self, features: Features) -> Features:
        output: Features = {}
        for angular_momentum, ts in features.items():
            # ModuleDict 取回來的靜態型別只到 Module，實際上是 ModuleList
            modules = cast(nn.ModuleList, self.layers[str(angular_momentum)])
            output[angular_momentum] = [m(t) for m, t in zip(modules, ts, strict=True)]
        return output


class SelfInteractionStep(_PerTensorStep):
    """對每條特徵做通道混合，全部混到同一個 ``output_dim``。

    L=0 用有 bias 的版本、L>0 用無 bias 的——理由見 :class:`SelfInteraction`。
    """

    def __init__(self, input_channels: Channels, output_dim: int) -> None:
        super().__init__(
            {
                angular_momentum: [
                    SelfInteraction(c, output_dim, bias=(angular_momentum == 0)) for c in cs
                ]
                for angular_momentum, cs in input_channels.items()
            }
        )
        self.output_channels: Channels = {
            angular_momentum: [output_dim] * len(cs)
            for angular_momentum, cs in input_channels.items()
        }


class NonlinearityStep(_PerTensorStep):
    """對每條特徵套用旋轉等變的非線性。通道數與 L 都不變。"""

    def __init__(
        self, input_channels: Channels, nonlin: Callable[[Tensor], Tensor] = F.elu
    ) -> None:
        super().__init__(
            {
                angular_momentum: [Nonlinearity(c, angular_momentum, nonlin=nonlin) for c in cs]
                for angular_momentum, cs in input_channels.items()
            }
        )
        self.output_channels: Channels = dict(input_channels)


class Layer(nn.Module):
    """論文 §5.1 的一個 module，四個步驟串起來。

    ``(Features, [N, N, rbf_count], [N, N, 3]) -> Features``

    :attr:`output_channels` 就是下一層的 ``input_channels``，票 07 靠這個疊三層。
    """

    def __init__(
        self,
        input_channels: Channels,
        rbf_count: int,
        output_dim: int,
        hidden_dim: int | None = None,
        radial_nonlin: Callable[[Tensor], Tensor] = F.relu,
        nonlin: Callable[[Tensor], Tensor] = F.elu,
    ) -> None:
        super().__init__()
        self.convolution = Convolution(
            input_channels, rbf_count, hidden_dim=hidden_dim, nonlin=radial_nonlin
        )
        self.self_interaction = SelfInteractionStep(
            concatenated_channels(self.convolution.output_channels), output_dim
        )
        self.nonlinearity = NonlinearityStep(self.self_interaction.output_channels, nonlin=nonlin)
        self.output_channels: Channels = self.nonlinearity.output_channels

    def forward(self, features: Features, rbf: Tensor, rij: Tensor) -> Features:
        features = self.convolution(features, rbf, rij)
        features = concatenation(features)
        features = self.self_interaction(features)
        return self.nonlinearity(features)
