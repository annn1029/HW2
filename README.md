# HW2：多元線性回歸（CRISP-DM）— 兩個資料集

🔗 **網頁（GitHub Pages）**：<https://annn1029.github.io/HW2/>（頁面上方的下拉選單可切換資料集 A／B）

| | 資料集 A：伊斯坦堡公寓房價 | 資料集 B：人機互動信任分數 |
|---|---|---|
| 來源 | [Istanbul Apartment Prices 2026](https://www.kaggle.com/datasets/brahimenesulusoy/istanbul-apartment-prices-2026)（brahimenesulusoy） | [Human-AI Interaction](https://www.kaggle.com/datasets/eshaalnmalik/human-ai-interaction)（eshaalnmalik） |
| 規模 | 24,767 筆、30 欄 | 10,100 筆、21 欄（20 個特徵） |
| 目標 | `price`（TRY，建模用 log） | `latent_trust_score`（約 38–100） |
| Notebook | [`HW2_MLR_CRISP-DM.ipynb`](HW2_MLR_CRISP-DM.ipynb) | [`HW2_AI_Trust_MLR.ipynb`](HW2_AI_Trust_MLR.ipynb) |
| 網頁 | `?ds=istanbul` | `?ds=trust` |

## 資料集 A 結果
| 模型 | 測試集 R²（log） | Adj R² | RMSE（log） | 價格 MedAPE | CV5 R² | 特徵數 |
|---|---|---|---|---|---|---|
| 簡單線性回歸（log_net_sqm） | 0.257 | 0.257 | 0.638 | 41.4% | 0.258 | 1 |
| 多元線性回歸（全特徵） | 0.803 | 0.800 | 0.329 | 19.5% | 0.800 | 70 |
| **多元線性回歸（Stepwise-AIC，最終模型）** | **0.803** | **0.800** | **0.329** | **19.5%** | **0.800** | **66** |
| 多元線性回歸（RFE 30） | 0.734 | 0.732 | 0.382 | 22.4% | 0.727 | 30 |
| LassoCV | 0.803 | 0.800 | 0.329 | 19.6% | – | 67 |

- 95% 預測區間在測試集的實際覆蓋率：**94.7%**
- 最重要的特徵：行政區 > 面積 > 屋齡 > 暖氣類型 > 產權 > 是否位於社區內

## 資料集 B 結果
| 模型 | 測試集 R² | Adj R² | RMSE | MAE | CV5 R² | 特徵數 |
|---|---|---|---|---|---|---|
| 簡單線性回歸（trust_history） | 0.217 | 0.216 | 7.94 | 6.34 | 0.227 | 1 |
| 多元線性回歸（全特徵） | 0.694 | 0.690 | 4.96 | 3.91 | 0.690 | 30 |
| **多元線性回歸（Stepwise-AIC，最終模型）** | **0.695** | **0.693** | **4.96** | **3.92** | **0.691** | **10** |
| 多元線性回歸（RFE 10） | 0.695 | 0.693 | 4.96 | 3.92 | 0.691 | 10 |
| LassoCV | 0.695 | 0.691 | 4.95 | 3.91 | – | 25 |

- 95% 預測區間在測試集的實際覆蓋率：**95.4%**
- Stepwise 從 20 個特徵中保留 9 個：信任歷史、介面複雜度、偏誤程度、回饋品質、模型準確率、認知負荷、錯誤率、推翻次數、法規嚴格度；年齡、教育、文化、領域、任務風險等全部被刪除
- `model_accuracy` 與 `error_rate` 高度共線（r = −0.97、VIF ≈ 16），兩者的個別係數不宜單獨解讀

## CRISP-DM（兩個資料集流程相同）
1. **Business Understanding**：定義預測目標、使用情境與成功標準
2. **Data Understanding**：缺值、分布、相關係數矩陣、類別箱型圖、散佈圖
3. **Data Preparation**：A：刪除洩漏欄位（`price_per_sqm`）、離群值、缺值旗標、衍生特徵、log 轉換；B：刪除重複值、修正負的錯誤率、**先切分再填補缺值**
4. **Modeling**：簡單線性回歸 → 全特徵多元線性回歸 → 特徵選擇（Forward Stepwise-AIC／RFE／LassoCV）
5. **Evaluation**：R²、Adj R²、RMSE、MAE、MAPE、5-fold CV、殘差診斷、VIF、**含 CI／PI 的預測圖**
6. **Deployment**：模型匯出為 JS，GitHub Pages 網頁上的預測器即時計算點預測與 95% 預測區間

## 檔案結構
```
├── HW2_MLR_CRISP-DM.ipynb      # 資料集 A 分析（已執行）
├── HW2_AI_Trust_MLR.ipynb      # 資料集 B 分析（已執行）
├── build_notebook.py / build_notebook_trust.py   # 產生 notebook 的腳本
├── data/
│   ├── istanbul_apartment_prices_2026.csv
│   └── human_ai_interaction.csv
├── docs/                       # GitHub Pages 根目錄
│   ├── index.html              # 報告 + 下拉選單 + 預測器
│   ├── model.js / model.json   # 資料集 A 模型
│   ├── assets/                 # 資料集 A 圖表
│   ├── trust/model.js          # 資料集 B 模型
│   ├── trust/assets/           # 資料集 B 圖表
│   └── chat_log.pdf            # AI 對話紀錄（自行匯出後放這裡）
└── requirements.txt
```

## 重現
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace HW2_MLR_CRISP-DM.ipynb
jupyter nbconvert --to notebook --execute --inplace HW2_AI_Trust_MLR.ipynb
```

## 部署到 GitHub Pages
Settings → Pages → Source 選 **Deploy from a branch**，Branch 選 `main`、資料夾選 `/docs` → Save。

## AI 協助說明
- **GPT 輔助內容**：程式架構、特徵工程、圖表與報告文字初稿由 AI 助理（Claude）協助產生，作者負責檢閱與執行；對話紀錄見 `docs/chat_log.pdf`。
- **NotebookLM 摘要**：由 Google NotebookLM 生成的「網路上同主題解法之研究摘要與比較」，全文見網頁「AI 協助紀錄」區塊（標示為 NotebookLM 生成）。
