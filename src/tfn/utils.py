"""幾何與數值工具。

移植對照：reference/tensorfieldnetworks-tf/tensorfieldnetworks/utils.py

上游的 rotation_equivariant_nonlinearity 也住在那支檔案裡，但它帶可學參數，
在 nn.Module 的設計下屬於 layers，不放這裡。
"""

import math

import numpy as np
import numpy.typing as npt
import scipy.linalg
import torch
import torch.nn.functional as F
from torch import Tensor

FLOAT_TYPE: torch.dtype = torch.float32
EPSILON: float = 1e-8

_LOG_2 = math.log(2.0)


def get_eijk(
    *, device: torch.device | str | None = None, dtype: torch.dtype = FLOAT_TYPE
) -> Tensor:
    """Levi-Civita 常數張量，shape ``[3, 3, 3]``。

    偶排列為 +1、奇排列為 -1、有重複指標為 0。它的用途是外積::

        (a × b)_i = ε_ijk a_j b_k

    這正是 1×1→1 卷積路徑的 Clebsch-Gordan 係數。
    """
    eijk = torch.zeros(3, 3, 3, device=device, dtype=dtype)
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1.0
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1.0
    return eijk


def norm_with_epsilon(
    x: Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
) -> Tensor:
    """正則化的 L2 範數：``sqrt(max(Σx², EPSILON))``。

    上游的參數叫 ``axis`` / ``keep_dims``（TF 慣例），這裡改用 torch 的
    ``dim`` / ``keepdim``。

    為什麼是「取 max」而不是「加上 epsilon」：``sqrt`` 在 0 點的導數未定義，
    取 max 之後零向量落在被夾住的區域，梯度是 0 而非 NaN。代價是零向量的
    範數回傳 ``sqrt(EPSILON)`` = 1e-4 而不是 0——需要真正的 0 時不能用這支。

    實作用 ``clamp`` 而非 ``torch.maximum``：後者兩個參數都必須是 Tensor，
    不吃 Python float，直接照 ``tf.maximum`` 翻會踩到。
    """
    squared = torch.sum(x * x, dim=dim, keepdim=keepdim)
    return torch.sqrt(torch.clamp(squared, min=EPSILON))


def ssp(x: Tensor) -> Tensor:
    """Shifted softplus：``ln(0.5·eˣ + 0.5)``。

    來自 SchNet (Schütt et al.)；論文 §5 說 radial function 與 nonlinearity
    與該文相同。減掉 ln2 讓它通過原點（一般 ``softplus(0) = ln2 ≈ 0.693``），
    且無限可微。

    寫成 ``softplus(x) - ln2``，代數上與上游的 ``log(0.5*exp(x)+0.5)`` 是同一個
    函數（``ln(1+eˣ) - ln2 = ln((1+eˣ)/2)``），差別在上游寫法會讓 float32 在
    ``x > 88.7`` 直接算出 inf。
    """
    return F.softplus(x) - _LOG_2


def difference_matrix(geometry: Tensor) -> Tensor:
    """相對位置矩陣：``rij[i, j] = r_i - r_j``。``[N, 3] -> [N, N, 3]``

    指標順序要緊：反過來會讓所有 L=1 特徵指向相反方向。
    """
    # [N, 1, 3] - [1, N, 3] -> [N, N, 3]
    return geometry.unsqueeze(1) - geometry.unsqueeze(0)


def distance_matrix(geometry: Tensor) -> Tensor:
    """相對距離矩陣。``[N, 3] -> [N, N]``

    走的是 :func:`norm_with_epsilon`，所以**對角線是 sqrt(EPSILON) = 1e-4
    而不是 0**。要判斷「同一個點」時不能拿這個結果去比 ``< EPSILON``。
    """
    return norm_with_epsilon(difference_matrix(geometry), dim=-1)


def rotation_matrix(
    axis: npt.ArrayLike,
    theta: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = FLOAT_TYPE,
) -> Tensor:
    """繞 ``axis`` 轉 ``theta`` 弧度的旋轉矩陣，shape ``[3, 3]``。

    慣例：``R @ v`` 把向量 v 依右手定則繞 axis 轉 theta。對 ``[N, 3]`` 的點雲
    要寫成 ``points @ R.T``，讓每一列各自被 R 作用。

    上游 notebook 寫的是 ``np.dot(shape, rotation)``，那實際上施加的是 R 的
    轉置。因為那裡的 R 是隨機的，統計上沒有差別；但拿來做確定性的等變性
    測試就會差，所以這裡把慣例講明。

    作法同上游：對反對稱矩陣取矩陣指數（Rodrigues 公式）。
    """
    axis_arr = np.asarray(axis, dtype=np.float64)
    # np.cross(eye(3), v) 逐列做 e_i × v，得到滿足 K·u = v × u 的反對稱矩陣
    skew = np.cross(np.eye(3), axis_arr * theta)
    return torch.as_tensor(np.asarray(scipy.linalg.expm(skew)), dtype=dtype, device=device)


def random_rotation_matrix(
    rng: int | np.random.Generator | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = FLOAT_TYPE,
) -> Tensor:
    """隨機的 3D 旋轉矩陣，shape ``[3, 3]``。

    ``rng`` 可以是 seed（int）、numpy ``Generator``、或 None（每次不同）。
    傳入 Generator 會推進它的狀態，所以連續呼叫得到不同的旋轉。

    上游收的是 numpy ``RandomState``，這裡改用較新的 ``Generator``。
    採樣方式同上游：軸向均勻分布於球面、角度均勻分布於 [0, 2π)。嚴格說這
    不是 SO(3) 上的 Haar 測度（那需要角度依 1-cos θ 加權），但支撐集涵蓋整個
    SO(3)，用來驗等變性與量測旋轉後的準確率都成立。
    """
    generator = np.random.default_rng(rng)
    axis = generator.standard_normal(3)
    axis = axis / (np.linalg.norm(axis) + EPSILON)
    theta = 2.0 * math.pi * generator.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta, device=device, dtype=dtype)
