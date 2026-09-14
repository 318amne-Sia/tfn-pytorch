"""notebook 的結構檢查。

三份 notebook 的第一格是同一份 Colab bootstrap，逐字複製。它要處理安裝、
「舊模組還在 sys.modules」、以及「本機 commit 了但沒 push」這幾件事，內容不短——
複製出來的東西會各自漂移，而且漂移完全沒有徵兆，直到某天只有一份 notebook 修好了。
這裡把它們釘在一起。

另外守著兩件會靜靜出錯的事：REQUIRES 裡的名字要真的存在（打錯的話會在 Colab 上
報「沒 push」，把人帶往錯的方向），以及存回去的 output 不能帶本機絕對路徑。
"""

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("*.ipynb"))
IDS = [path.name for path in NOTEBOOKS]


def cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def bootstrap_source(path: Path) -> str:
    cell = cells(path)[0]
    assert cell["cell_type"] == "code", f"{path.name} 的第一格不是程式碼"
    source = "".join(cell["source"])
    assert "Colab bootstrap" in source, f"{path.name} 的第一格不是 bootstrap"
    return source


def split_requires(path: Path) -> tuple[str, list[str]]:
    """把 bootstrap 拆成「REQUIRES 以外的部分」與「REQUIRES 的內容」。"""
    kept: list[str] = []
    literal: list[str] = []
    inside = False
    for line in bootstrap_source(path).splitlines():
        if line.startswith("REQUIRES = ["):
            inside = True
            literal.append(line.removeprefix("REQUIRES = "))
            continue
        if inside:
            literal.append(line)
            if line == "]":
                inside = False
            continue
        kept.append(line)
    assert not inside, f"{path.name} 的 REQUIRES 沒有正常結束"
    assert literal, f"{path.name} 找不到 REQUIRES"
    return "\n".join(kept), ast.literal_eval("\n".join(literal))


def test_there_are_three_notebooks():
    assert IDS == ["gravity.ipynb", "moment_of_inertia.ipynb", "shape_classification.ipynb"]


def test_every_bootstrap_is_identical_apart_from_requires():
    """三份只能差在 REQUIRES，其餘逐字相同。

    失敗的話不要「就地修一份」——改的是所有 notebook 的同一格，三份要一起改。
    """
    bodies = {path.name: split_requires(path)[0] for path in NOTEBOOKS}
    reference = bodies[IDS[0]]
    for name, body in bodies.items():
        assert body == reference, f"{name} 的 bootstrap 與 {IDS[0]} 不一致"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=IDS)
def test_every_required_entry_actually_exists(path: Path):
    """REQUIRES 裡的名字打錯的話，Colab 上會報成「沒 push」——完全錯的方向。"""
    for entry in split_requires(path)[1]:
        module, _, name = entry.partition(":")
        imported = importlib.import_module(module)
        if name:
            assert hasattr(imported, name), f"{module} 沒有 {name}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=IDS)
def test_the_notebook_requires_its_own_experiment_module(path: Path):
    """每份 notebook 至少要檢查自己那個實驗的模組。"""
    expected = f"tfn.{path.stem}"
    assert expected in split_requires(path)[1], f"REQUIRES 少了 {expected}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=IDS)
def test_committed_outputs_carry_no_local_paths(path: Path):
    """存回去的 output 不該把本機絕對路徑帶進一個 public repo。

    bootstrap 會印出 tfn.__file__，本機跑的話那是 /Users/<名字>/... 開頭。
    """
    outputs = json.dumps([cell.get("outputs", []) for cell in cells(path)], ensure_ascii=False)
    for prefix in ("/Users/", "/home/", "C:\\Users"):
        assert prefix not in outputs, f"{path.name} 的 output 裡有 {prefix}"
