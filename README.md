# HW2：伊斯坦堡公寓房價多元線性回歸（CRISP-DM）

🔗 **網頁（GitHub Pages）**：<https://annn1029.github.io/HW2/>
📓 **Notebook**：[`HW2_MLR_CRISP-DM.ipynb`](HW2_MLR_CRISP-DM.ipynb)

## 資料集
- **Istanbul Apartment Prices 2026**（Kaggle，作者 brahimenesulusoy）
  <https://www.kaggle.com/datasets/brahimenesulusoy/istanbul-apartment-prices-2026>
- 24,767 筆、30 欄（數值與類別特徵 > 13 個），目標變數為 `price`（TRY）

## 結果摘要
| 模型 | 測試集 R²（log） | Adj R² | RMSE（log） | 價格 MedAPE | CV5 R² | 特徵數 |
|---|---|---|---|---|---|---|
| 簡單線性回歸（log_net_sqm） | 0.257 | 0.257 | 0.638 | 41.4% | 0.258 | 1 |
| 多元線性回歸（全特徵） | 0.803 | 0.800 | 0.329 | 19.5% | 0.800 | 70 |
| **多元線性回歸（Stepwise-AIC，最終模型）** | **0.803** | **0.800** | **0.329** | **19.5%** | **0.800** | **66** |
| 多元線性回歸（RFE 30） | 0.734 | 0.732 | 0.382 | 22.4% | 0.727 | 30 |
| LassoCV | 0.803 | 0.800 | 0.329 | 19.6% | – | 67 |

- 95% 預測區間在測試集的實際覆蓋率：**94.7%**
- 最重要的特徵：行政區 > 面積 > 屋齡 > 暖氣類型 > 產權 > 是否位於社區內

## CRISP-DM
1. **Business Understanding**：為伊斯坦堡公寓提供可解釋的估價模型，並以預測區間表達不確定性
2. **Data Understanding**：缺值分析、價格分布、相關係數矩陣、行政區箱型圖
3. **Data Preparation**：刪除洩漏欄位（`price_per_sqm`）、處理離群值、填補缺值並加旗標、衍生特徵、one-hot 編碼、log 轉換
4. **Modeling**：簡單線性回歸 → 全特徵多元線性回歸 → 特徵選擇（Forward Stepwise-AIC／RFE／LassoCV）
5. **Evaluation**：R²、Adj R²、RMSE、MAE、MAPE、5-fold CV、殘差診斷、VIF、**含 CI／PI 的預測圖**
6. **Deployment**：模型匯出為 `docs/model.js`，GitHub Pages 網頁上的 JavaScript 估價器即時計算點預測與 95% 預測區間

## 檔案結構
```
├── HW2_MLR_CRISP-DM.ipynb   # 主分析（已執行，含所有輸出）
├── build_notebook.py        # 產生 notebook 的腳本
├── data/istanbul_apartment_prices_2026.csv
├── docs/                    # GitHub Pages 根目錄
│   ├── index.html           # 報告 + 估價器
│   ├── model.js / model.json
│   ├── assets/              # 圖表（PNG）與結果表（CSV）
│   └── chat_log.pdf         # AI 對話紀錄（自行匯出後放這裡）
└── requirements.txt
```

## 重現
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace HW2_MLR_CRISP-DM.ipynb
```

## 部署到 GitHub Pages
Settings → Pages → Source 選 **Deploy from a branch**，Branch 選 `main`、資料夾選 `/docs` → Save。

## AI 協助說明
- **GPT 輔助內容**：程式架構、特徵工程、圖表與報告文字初稿由 AI 助理（Claude）協助產生，作者負責檢閱與執行；對話紀錄見 `docs/chat_log.pdf`。
- **NotebookLM 摘要**：見網頁中的「AI 協助紀錄」區塊。
