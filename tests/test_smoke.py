"""Smoke tests：套件裝得起來，而且底下的 torch 真的能動。

這些測試刻意不碰任何 TFN 數學——它們守的是環境與打包，不是模型。
真正的等變性測試從票 02 開始才會出現。
"""

import tfn


def test_package_imports_with_version():
    """src layout 的套件確實被安裝、且版本號讀得到。

    在 Colab 上這一條就是 bootstrap 成功與否的判準。
    """
    assert isinstance(tfn.__version__, str)
    assert tfn.__version__


def test_torch_is_usable():
    """torch 裝好且能實際算東西。

    Colab 那端我們用 --no-deps 安裝，吃的是它預裝的 torch，
    所以「torch 到底能不能動」值得獨立驗一條。
    """
    import torch

    x = torch.ones(2, 3)
    assert x.sum().item() == 6.0
