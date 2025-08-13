# radar-gesture-recognition

以毫米波雷達 **feature map / raw** 訊號進行**手勢辨識**。目前功能聚焦在：

1. 資料**前處理**（`.h5` → `.npz`）
2. **訓練** 3D CNN
3. **評估**（混淆矩陣）
4. **線上即時推論（GUI）**

---

## Kaggle 資料集

- Kaggle dataset：https://www.kaggle.com/datasets/shengkai1020/handgesture-mmwave-v1-0?select=HandGesture-mmWave-v1.0

### 資料格式

每個 `.h5` 檔至少包含：

- `DS1`: 雷達特徵序列  
  - 形狀： `(2, 32, 32, T)`  
- `LABEL`: 長度 `T` 的 0/1（按住手勢=1、放開=0），`uint8`

資料夾結構（依類別分資料夾；順序需與模型輸出一致：`["Background","PatPat","Wave","Come"]`：
- `data/`
  - `train/`
    - `background/` — `*.h5`
    - `patpat/` — `*.h5`
    - `wave/` — `*.h5`
    - `come/` — `*.h5`
  - `val/`
    - `background/` — `*.h5`
    - `patpat/` — `*.h5`
    - `wave/` — `*.h5`
    - `come/` — `*.h5`
    - 
### 安裝環境

- `.\setup.bat`
