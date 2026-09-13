"""實驗一（3D Tetris 形狀分類）的測試。

移植對照：reference/tensorfieldnetworks-tf/shape_classification.ipynb

這裡最重要的一條是 test_untrained_model_is_invariant_to_pose：模型在**還沒
訓練**時就該對旋轉與平移免疫。等變性是結構帶來的，不是學來的——如果這條
不過，後面那條「訓練到 100%」就算僥倖過了也不算數。
"""

from typing import cast

import numpy as np
import pytest
import torch

from tfn import utils
from tfn.layers import Layer
from tfn.shape_classification import (
    SHAPE_NAMES,
    TETRIS,
    Evaluation,
    ShapeClassifier,
    evaluate,
    random_pose,
    tetris_shapes,
    train,
)

# 論文宣稱的 perfect accuracy 需要這麼多 epoch：實測 5 個 seed 在 600 epochs
# 全部到 100%，400 會漏掉其中一個。
EPOCHS_TO_CONVERGE = 600


# --------------------------------------------------------------------------
# 資料集
# --------------------------------------------------------------------------


def test_dataset_has_eight_four_point_shapes():
    shapes = tetris_shapes()
    assert len(shapes) == len(SHAPE_NAMES) == 8
    assert all(shape.shape == (4, 3) for shape in shapes)


def test_chiral_shapes_are_indistinguishable_by_distance_alone():
    """兩個鏡像形狀的點對距離集合完全相同。

    這是本實驗的主張所在：只看距離的模型（SchNet）在這兩個形狀上必定
    失敗，因為它們餵給模型的輸入根本一模一樣。TFN 能分開，靠的是 L=1
    特徵帶的方向資訊。
    """
    first, second = tetris_shapes()[:2]
    assert SHAPE_NAMES[:2] == ("chiral_shape_1", "chiral_shape_2")
    assert torch.allclose(
        torch.sort(utils.distance_matrix(first).flatten()).values,
        torch.sort(utils.distance_matrix(second).flatten()).values,
        atol=1e-5,
    )


def test_random_pose_is_a_rigid_motion():
    """隨機姿態只能轉與移，不能改變形狀本身。"""
    shape = tetris_shapes()[4]
    posed = random_pose(shape, np.random.default_rng(0))
    assert torch.allclose(utils.distance_matrix(posed), utils.distance_matrix(shape), atol=1e-5)


def test_random_pose_actually_moves_the_shape():
    """反面對照：確認上一條不是因為 random_pose 什麼都沒做而通過。"""
    shape = tetris_shapes()[4]
    assert not torch.allclose(shape, random_pose(shape, np.random.default_rng(0)), atol=1e-3)


def test_random_pose_translates_as_well_as_rotates():
    """上游 notebook 算了 translated_shape 卻餵 rotated_shape，平移那半從沒被測到。

    質心搬走了就代表平移真的施加了——純旋轉會讓質心繞著原點轉，但這裡的
    形狀質心離原點很近，平移量在 ±3，兩者量級分得開。
    """
    shape = tetris_shapes()[3]
    centroids = torch.stack(
        [random_pose(shape, np.random.default_rng(seed)).mean(dim=0) for seed in range(8)]
    )
    assert centroids.norm(dim=-1).max() > shape.mean(dim=0).norm() + 1.0


def test_random_pose_is_reproducible_from_a_seed():
    shape = tetris_shapes()[0]
    first = random_pose(shape, np.random.default_rng(7))
    second = random_pose(shape, np.random.default_rng(7))
    assert torch.equal(first, second)


# --------------------------------------------------------------------------
# 模型結構
# --------------------------------------------------------------------------


def test_model_outputs_one_logit_per_class():
    model = ShapeClassifier(len(TETRIS))
    assert model(tetris_shapes()[0]).shape == (len(TETRIS),)


def test_model_stacks_three_layers_and_reads_out_scalars():
    """論文 §5.1：layer_dims = [1, 4, 4, 4]，三個 module，只取 L=0 輸出。"""
    model = ShapeClassifier(len(TETRIS))
    assert len(model.blocks) == 3
    # ModuleList 取回來的靜態型別只到 Module
    assert cast(Layer, model.blocks[-1]).output_channels[0] == [4]
    assert model.readout.in_features == 4
    assert model.readout.bias is not None


def test_model_registers_every_submodule():
    """embed、三層、readout 的參數都要被 optimizer 看得到。"""
    model = ShapeClassifier(len(TETRIS))
    by_hand = len(list(model.embed.parameters())) + len(list(model.readout.parameters()))
    for block in model.blocks:
        by_hand += len(list(block.parameters()))
    assert len(list(model.parameters())) == by_hand


def test_untrained_model_is_invariant_to_pose():
    """還沒訓練就該對旋轉與平移免疫——等變性是結構帶來的，不是學來的。

    整條鏈（票 02 到 06）如果有任何一環的等變性是假的，這裡就會露餡，
    而且不必等到訓練完。
    """
    torch.manual_seed(0)
    model = ShapeClassifier(len(TETRIS))
    for shape in tetris_shapes():
        posed = random_pose(shape, np.random.default_rng(1))
        assert torch.allclose(model(posed), model(shape), atol=1e-5)


def test_untrained_model_separates_the_chiral_pair():
    """兩個鏡像形狀餵出來的 logits 不同——距離集合相同也擋不住。"""
    torch.manual_seed(0)
    model = ShapeClassifier(len(TETRIS))
    first, second = tetris_shapes()[:2]
    assert not torch.allclose(model(first), model(second), atol=1e-4)


# --------------------------------------------------------------------------
# 訓練
# --------------------------------------------------------------------------


def test_training_is_reproducible_from_a_seed():
    """同一個 seed，兩次訓練的 loss 軌跡完全相同。"""

    def run() -> list[float]:
        torch.manual_seed(0)
        return train(ShapeClassifier(len(TETRIS)), tetris_shapes(), epochs=20)

    assert run() == run()


def test_training_reduces_the_loss():
    torch.manual_seed(0)
    history = train(ShapeClassifier(len(TETRIS)), tetris_shapes(), epochs=30)
    assert history[-1] < history[0]


def test_evaluation_is_reproducible_from_a_seed():
    torch.manual_seed(0)
    model = ShapeClassifier(len(TETRIS))
    shapes = tetris_shapes()
    assert evaluate(model, shapes, rounds=3, rng=5) == evaluate(model, shapes, rounds=3, rng=5)


def test_evaluation_counts_every_sample():
    torch.manual_seed(0)
    result = evaluate(ShapeClassifier(len(TETRIS)), tetris_shapes(), rounds=4, rng=0)
    assert all(len(guesses) == 4 for guesses in result.predictions.values())
    assert set(result.predictions) == set(SHAPE_NAMES)


def test_accuracy_for_reads_the_diagonal():
    result = Evaluation(0.5, {"a": ["a", "b"], "b": ["b", "b"]})
    assert result.accuracy_for("a") == 0.5
    assert result.accuracy_for("b") == 1.0


# --------------------------------------------------------------------------
# 兌現點
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_run() -> Evaluation:
    """訓練一次、測一次，給下面兩條慢測試共用。

    訓練只餵單一朝向、無資料增強；測試集全部隨機旋轉且平移。
    """
    torch.manual_seed(0)
    model = ShapeClassifier(len(TETRIS))
    shapes = tetris_shapes()
    train(model, shapes, epochs=EPOCHS_TO_CONVERGE)
    return evaluate(model, shapes, rounds=25, rng=0)


@pytest.mark.slow
def test_trains_to_perfect_accuracy_on_unseen_poses(trained_run: Evaluation):
    """整條鏈的兌現點：只餵單一朝向訓練，測試集全部隨機旋轉且平移，仍然全對。

    200 個樣本（25 輪 × 8 形狀），沒有一個是訓練時看過的姿態。
    """
    assert sum(len(guesses) for guesses in trained_run.predictions.values()) == 200
    per_shape = {name: trained_run.accuracy_for(name) for name in SHAPE_NAMES}
    assert trained_run.accuracy == 1.0, f"每個形狀的準確率：{per_shape}"


@pytest.mark.slow
def test_the_chiral_pair_is_never_confused(trained_run: Evaluation):
    """論文對本實驗的主要主張：chiral_shape_1 / chiral_shape_2 不會互相混淆。"""
    assert "chiral_shape_2" not in trained_run.predictions["chiral_shape_1"]
    assert "chiral_shape_1" not in trained_run.predictions["chiral_shape_2"]
    assert trained_run.accuracy_for("chiral_shape_1") == 1.0
    assert trained_run.accuracy_for("chiral_shape_2") == 1.0
