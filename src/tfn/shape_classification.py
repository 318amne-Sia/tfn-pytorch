"""實驗一：3D Tetris 形狀分類（論文 §5.1）。

移植對照：reference/tensorfieldnetworks-tf/shape_classification.ipynb

論文的主張是：訓練只餵單一朝向、完全不做旋轉資料增強，測試時餵隨機旋轉且
平移過的同一批形狀，仍然全對。這不是靠模型「學會」旋轉，而是等變性讓它
根本不需要學——網路內部的表示會跟著座標一起轉。

兩個鏡像形狀（chiral_shape_1 / chiral_shape_2）是本實驗的重點：它們的
點對距離集合完全相同，只靠距離的模型（SchNet）分不出來；只加上角度也不夠
（ANI-1）。要分辨它們需要能感知手性的表示。
"""

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tfn.layers import Channels, Features, Layer, SelfInteraction
from tfn.utils import (
    FLOAT_TYPE,
    difference_matrix,
    distance_matrix,
    random_rotation_matrix,
    rbf_expansion,
)

__all__ = [
    "TETRIS",
    "SHAPE_NAMES",
    "RBF_LOW",
    "RBF_HIGH",
    "RBF_COUNT",
    "LAYER_DIMS",
    "tetris_shapes",
    "random_pose",
    "ShapeClassifier",
    "train",
    "evaluate",
    "Evaluation",
]

# 8 個形狀，順序即 label。同上游 notebook。
TETRIS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "chiral_shape_1": ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)),
    "chiral_shape_2": ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)),
    "square": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
    "line": ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)),
    "corner": ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)),
    "T": ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)),
    "zigzag": ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)),
    "L": ((0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)),
}
SHAPE_NAMES: tuple[str, ...] = tuple(TETRIS)

# RBF 設定與網路寬度，同上游 notebook
RBF_LOW = 0.0
RBF_HIGH = 3.5
RBF_COUNT = 4
LAYER_DIMS: tuple[int, ...] = (1, 4, 4, 4)


def tetris_shapes(
    *, device: torch.device | str | None = None, dtype: torch.dtype = FLOAT_TYPE
) -> list[Tensor]:
    """8 個形狀的座標，每個 ``[4, 3]``。索引即 label。"""
    return [torch.tensor(points, device=device, dtype=dtype) for points in TETRIS.values()]


def random_pose(shape: Tensor, rng: np.random.Generator) -> Tensor:
    """隨機旋轉**並平移**一個形狀。

    上游 notebook 在這裡有個 bug：算了 ``translated_shape`` 卻把
    ``rotated_shape`` 餵進去，所以平移那半從來沒被測到。這裡修掉——反正
    整層網路本來就該對平移免疫，測到才有意義。
    """
    rotation = random_rotation_matrix(rng, device=shape.device, dtype=shape.dtype)
    translation = torch.as_tensor(
        rng.uniform(low=-3.0, high=3.0, size=3), device=shape.device, dtype=shape.dtype
    )
    return shape @ rotation.T + translation


class ShapeClassifier(nn.Module):
    """論文 §5.1 的網路。``[N, 3] -> [num_classes]``

    結構::

        embed（無 bias 的 self-interaction，把常數 1 攤成 layer_dims[0] 個通道）
          → Layer × 3
          → 取 L=0 的輸出、對所有點取平均（全域池化）
          → 全連接層 + bias

    起點是全 1 的 L=0 特徵：這個實驗裡所有點都是同一種「原子」，沒有元素
    資訊可餵，所以形狀的全部訊息都來自幾何。

    只取 L=0 輸出，是因為分類結果必須是旋轉不變的純量；L=1 那半在最後一層
    之後就丟掉了，但它一路上都在替 L=0 那半搬運方向資訊。

    一次吃一個形狀，不 batch——同作者，而且 N=4 的點雲小到 batch 沒有意義。
    """

    def __init__(
        self,
        num_classes: int,
        layer_dims: tuple[int, ...] = LAYER_DIMS,
        rbf_count: int = RBF_COUNT,
        rbf_low: float = RBF_LOW,
        rbf_high: float = RBF_HIGH,
    ) -> None:
        super().__init__()
        self.rbf_count = rbf_count
        self.rbf_low = rbf_low
        self.rbf_high = rbf_high

        self.embed = SelfInteraction(1, layer_dims[0], bias=False)
        channels: Channels = {0: [layer_dims[0]]}
        blocks = []
        for output_dim in layer_dims[1:]:
            block = Layer(channels, rbf_count, output_dim)
            blocks.append(block)
            channels = block.output_channels
        self.blocks = nn.ModuleList(blocks)

        self.readout = nn.Linear(channels[0][0], num_classes)
        # 上游用 tf.get_variable 的預設（glorot uniform）；bias 這裡取 0
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, geometry: Tensor) -> Tensor:
        rij = difference_matrix(geometry)
        rbf = rbf_expansion(
            distance_matrix(geometry), low=self.rbf_low, high=self.rbf_high, count=self.rbf_count
        )

        # 常數輸入跟著**參數**的 device / dtype，不是跟著 geometry：模型與輸入
        # dtype 不合時，錯誤要出在 geometry 進來的地方，而不是被這個張量掩護到
        # 更裡面才炸
        weight = self.embed.linear.weight
        ones = torch.ones(geometry.shape[0], 1, 1, device=weight.device, dtype=weight.dtype)
        features: Features = {0: [self.embed(ones)]}
        for block in self.blocks:
            features = block(features, rbf, rij)

        # [N, C, 1] -> [N, C] -> [C]。只 squeeze 最後一軸：上游用 tf.squeeze
        # 會連 N 一起吃掉，單點輸入時就壞了
        pooled = features[0][0].squeeze(-1).mean(dim=0)
        return self.readout(pooled)


def train(
    model: ShapeClassifier,
    shapes: list[Tensor],
    *,
    epochs: int,
    learning_rate: float = 1e-3,
) -> list[float]:
    """訓練，回傳每個 epoch 的平均 loss。

    只餵單一朝向、不做任何資料增強——這正是論文要證明的事：等變性讓模型
    不需要看過旋轉樣本。一個 epoch 走完 8 個形狀，每個形狀各更新一次。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        total = 0.0
        for label, shape in enumerate(shapes):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(shape), torch.tensor(label, device=shape.device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        history.append(total / len(shapes))
    return history


class Evaluation(NamedTuple):
    """測試結果。``predictions[真實形狀名]`` 是每一輪被判成什麼。"""

    accuracy: float
    predictions: dict[str, list[str]]

    def accuracy_for(self, name: str) -> float:
        guesses = self.predictions[name]
        if not guesses:
            raise ValueError(f"{name} 沒有任何預測，算不出準確率")
        return sum(guess == name for guess in guesses) / len(guesses)


@torch.no_grad()
def evaluate(
    model: ShapeClassifier,
    shapes: list[Tensor],
    *,
    rounds: int = 25,
    rng: np.random.Generator | int | None = None,
    names: tuple[str, ...] = SHAPE_NAMES,
) -> Evaluation:
    """在隨機旋轉且平移過的形狀上量準確率。

    每一輪替每個形狀抽一組新的姿態，所以 ``rounds`` 輪 × 8 個形狀 =
    ``8 * rounds`` 個樣本。傳 int 或 Generator 給 ``rng`` 就可重現。
    """
    if len(shapes) != len(names):
        raise ValueError(f"shapes 與 names 數量不符：{len(shapes)} 個形狀、{len(names)} 個名字")
    if rounds < 1:
        raise ValueError(f"rounds 至少要 1，收到 {rounds}")

    generator = np.random.default_rng(rng)
    predictions: dict[str, list[str]] = {name: [] for name in names}
    correct = 0
    for _ in range(rounds):
        for label, shape in enumerate(shapes):
            guess = int(model(random_pose(shape, generator)).argmax())
            predictions[names[label]].append(names[guess])
            correct += guess == label
    return Evaluation(correct / (rounds * len(shapes)), predictions)
