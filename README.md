# tfn-pytorch

[Tensor Field Networks](https://arxiv.org/abs/1802.08219) (Thomas et al., 2018) 的 PyTorch 移植，
用來一個一個復現論文的三個實驗。

上游原作是 TensorFlow 1.x（graph mode、Python 2），現在已經跑不起來：
TF 1.15 最高只支援 Python 3.7，macOS arm64 根本沒有 wheel，Colab 也早就拔掉了 `%tensorflow_version 1.x`。

## 進度

| 實驗 | 論文章節 | 狀態 |
| --- | --- | --- |
| shape classification（3D Tetris） | §5.1 | 完成——測試準確率 100% |
| Newtonian gravity | §5.2 | 完成——徑向函數對 `−1/r²` 的 nRMSE 0.099 |
| moment of inertia | §5.2 | 進行中 |
| missing point（QM9） | §5.3 | 未開始 |

## 在 Colab 上用

Colab runtime 看不到你本機的檔案，所以套件必須從 GitHub 抓。把這格放在 notebook 第一格：

```python
%pip install -q --force-reinstall --no-deps \
    git+https://github.com/318amne-Sia/tfn-pytorch.git@main

import tfn
print(tfn.__version__)
```

兩個旗標都不是可有可無的：

- **`--no-deps`** —— Colab 已經預裝 torch / numpy / scipy，而且那個 torch 是對著它自己的 CUDA 編的。
  少了這個旗標，pip 會看到我們宣告的 `torch>=2.0` 而跑去 PyPI 重裝一份通用版：
  下載近 1 GB、花好幾分鐘，還可能把 GPU 弄丟。
- **`--force-reinstall`** —— 版本號沒變時 pip 會直接跳過安裝。推了修正之後想抓到新版就得靠它。

重裝之後舊模組還在 `sys.modules` 裡，**要重啟 kernel** 才吃得到新版。

Colab 的 pip 安裝不持久，換一台 VM 就沒了，所以這格每個 session 都要跑一次。

## 在本機開發

實驗一（N=4、batch=1）在 CPU 上幾分鐘就跑完，丟 GPU 只會被 kernel launch 開銷拖慢，
所以寫程式和跑測試都在本機做，Colab 只用來驗收整條管線。

```sh
uv sync                        # 建 Python 3.12 環境（對齊 Colab 的 3.12.13）並安裝
uv run pytest                  # 測試
uv run pytest -m "not slow"    # 跳過訓練到收斂的那幾條（13 秒 -> 2 秒）
uv run ruff check .            # lint
uv run ruff format .           # 格式化
uv run pyright                 # 型別檢查
```

`ruff`、`pyright`、`matplotlib` 都只在本機開發時用，不是執行期依賴，
所以 Colab 那端的 `--no-deps` 安裝完全不受影響。
（Colab 本來就預裝 matplotlib；它在 dev 這組，是為了讓 notebook 的畫圖程式碼
能先在本機驗過再放進去。）

它們抓的是**機械性翻譯錯誤**（API 簽名不合、殘留 import、`is` 比較字面值），
不是數學錯誤——einsum 索引排錯或 CG 符號弄反，靜態工具一律看不見，
那是 `tests/` 裡等變性測試的職責。

套件採 src layout：程式碼在 `src/tfn/`，不在 repo 根目錄。
這逼得本機測試 import 到的一定是「安裝後」的那一份，跟 Colab 上的情況一致，
避免「本機跑得動、`pip install` 之後壞掉」。

## 目錄

| 路徑 | 內容 |
| --- | --- |
| `src/tfn/` | 移植後的套件 |
| `tests/` | 測試。核心是等變性測試：旋轉輸入後 L=0 輸出不變、L=1 輸出跟著轉 |
| `notebooks/` | 各實驗的 notebook |
| `reference/` | 作者原始 TF1 實作，**唯讀對照用，不要改** |

## 實驗一：3D Tetris 形狀分類

`notebooks/shape_classification.ipynb`（本機約 20 秒，Colab 同樣跑得完）。

論文 §5.1 的主張是：訓練只餵單一朝向、完全不做旋轉資料增強，測試時餵隨機旋轉
**且平移**過的同一批形狀，仍然全部分對。復現結果：200 個樣本（25 輪 × 8 形狀）
準確率 **100%**，兩個鏡像形狀 `chiral_shape_1` / `chiral_shape_2` 零混淆。

等變性是結構帶來的，不是訓練出來的——notebook 裡有一格驗證**還沒訓練**的模型
對旋轉加平移的 logits 偏差就只有 1e-7。

與上游 notebook 的兩處差異：

- 上游測試迴圈算了 `translated_shape` 卻把 `rotated_shape` 餵進去，平移那半
  從來沒被測到。這裡修掉了。
- epoch 數取 1000（上游 2001）。實測 5 個不同 seed 在 600 epochs 就全部到 100%。

## reference/ 是什麼

`reference/tensorfieldnetworks-tf/` 是[作者原始 repo](https://github.com/tensorfieldnetworks/tensorfieldnetworks) 的副本
（commit `6850fd6`, 2020-01-07, MIT）。移植時逐行對照用，出處與 commit 記在 `reference/README.md`。

論文 PDF 不進 repo（見 `.gitignore`），從 [arXiv](https://arxiv.org/abs/1802.08219) 取得。

## 授權

MIT，沿用上游。見 `LICENSE`。
