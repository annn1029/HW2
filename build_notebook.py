"""Build HW2_MLR_CRISP-DM.ipynb from cell definitions (run once, then execute with nbconvert)."""
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# HW2：伊斯坦堡公寓房價之多元線性回歸分析（CRISP-DM）

- **資料集**：Istanbul Apartment Prices 2026（Kaggle, brahimenesulusoy）
  <https://www.kaggle.com/datasets/brahimenesulusoy/istanbul-apartment-prices-2026>
- **筆數 / 欄位**：24,767 筆、30 欄（可用特徵 > 13 個）
- **目標變數**：`price`（土耳其里拉 TRY），建模時使用 `log(price)`
- **方法**：簡單線性回歸（基準）→ 多元線性回歸 → 特徵選擇（Forward Stepwise-AIC、RFE、LassoCV）→ 評估 → 預測區間 → 部署（GitHub Pages）

> 🤖 **GPT 輔助內容**：本 notebook 之程式架構、特徵工程與文字說明由 AI 助理（Claude）協助產生，經作者檢閱與執行。
""")

code(r"""
import warnings, json, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RNG = 42
FIG = "docs/assets"; os.makedirs(FIG, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight"})
ACC, ACC2, GREY = "#2563eb", "#f97316", "#64748b"
def save(name): plt.savefig(f"{FIG}/{name}.png"); plt.show()
""")

md(r"""
## 1. Business Understanding（商業理解）

**問題**：伊斯坦堡房價在高通膨下波動劇烈，買方、賣方與仲介需要一個**透明、可解釋**的估價工具。

**目標**：以房屋的面積、房間數、樓層、屋齡、行政區、暖氣、產權等屬性，建立線性回歸模型預測公寓售價，並提供**預測區間**表達不確定性。

**成功標準**
- 測試集 R²（log 尺度）≥ 0.75
- 95% 預測區間的實際覆蓋率接近 95%
- 模型係數可解釋（例如「面積增加 1%，價格約增加 β%」）

**選用線性回歸的理由**：係數可直接解讀為各屬性的邊際價值；搭配 log 轉換後，係數代表百分比變化，適合房價這類右偏資料。
""")

md("## 2. Data Understanding（資料理解）")
code(r"""
raw = pd.read_csv("data/istanbul_apartment_prices_2026.csv", encoding="utf-8-sig")
print(raw.shape)
raw.head()
""")
code(r"""
info = pd.DataFrame({"dtype": raw.dtypes.astype(str),
                     "missing_%": (raw.isna().mean()*100).round(1),
                     "n_unique": raw.nunique()})
info.sort_values("missing_%", ascending=False)
""")
code(r"""
raw[["price","price_per_sqm","gross_sqm","net_sqm","rooms","total_floors","building_age","bathroom_count"]].describe(
    percentiles=[.01,.5,.99]).T.round(1)
""")
md(r"""
**觀察**
1. `price` 極度右偏（中位數約 675 萬，最大值 115 億），存在明顯輸入錯誤 → 需處理離群值並取 log。
2. `price_per_sqm = price / gross_sqm` 是由目標推導出的欄位 → **資料洩漏 (data leakage)，必須刪除**。
3. `complex_name`、`floor_category`、`maintenance_fee`、`building_type`、`fuel_type`、`building_condition` 缺值 > 50%。
4. `rooms`=114、`total_floors`=450、`building_age`=400 等為不合理值。
""")
code(r"""
missing = raw.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]*100
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(missing.index[::-1], missing.values[::-1], color=ACC)
ax.set_xlabel("Missing (%)"); ax.set_title("Missing values by column")
save("01_missing")
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
p = raw["price"]; p = p[p.between(p.quantile(.005), p.quantile(.995))]
axes[0].hist(p/1e6, bins=80, color=GREY); axes[0].set_title("Price (million TRY)"); axes[0].set_xlabel("price")
axes[1].hist(np.log(p), bins=80, color=ACC); axes[1].set_title("log(price) — closer to normal"); axes[1].set_xlabel("log(price)")
save("02_price_dist")
""")

md("## 3. Data Preparation（資料準備）")
md(r"""
**清理規則**
| 步驟 | 規則 | 理由 |
|---|---|---|
| 刪除洩漏 / 無用欄位 | `price_per_sqm`, `listing_id`, `last_updated`, `scraped_at`, `complex_name`, `neighborhood`(583 類) | 洩漏、ID、高基數 |
| 價格離群值 | 保留 0.5%–99.5% 分位數 | 去除輸入錯誤與極端豪宅 |
| 合理範圍 | 25 ≤ net_sqm ≤ 500、net ≤ gross、rooms ≤ 10、total_floors ≤ 60、building_age ≤ 100、floor ≤ total_floors | 物理上不合理 |
| 缺值 | 數值以中位數填補並加「是否缺值」旗標；類別補 `Missing` | 保留資訊而不丟樣本 |
| 衍生特徵 | `log_net_sqm`、`faces_south`、`n_directions`、`has_maint_fee`、`floor_ratio` | 捕捉非線性 / 方位價值 |
| 類別編碼 | One-hot（drop_first）；稀有類別併為 `Other` | 線性模型需數值輸入 |
""")
code(r"""
df = raw.drop(columns=["price_per_sqm","listing_id","last_updated","scraped_at","complex_name","neighborhood"]).copy()
n0 = len(df)
lo, hi = df.price.quantile([.005, .995])
df = df[df.price.between(lo, hi)]
df = df[df.net_sqm.between(25, 500) & (df.net_sqm <= df.gross_sqm) & (df.rooms.between(1, 10))
        & (df.total_floors.between(1, 60)) & (df.building_age.between(0, 100))]
df = df[df.floor.isna() | (df.floor <= df.total_floors)]
df = df[df.district != "Çatalca"]   # only 7 listings
print(f"rows: {n0} -> {len(df)}  (removed {n0-len(df)}, {100*(n0-len(df))/n0:.1f}%)")
""")
code(r"""
def lump(s, keep):
    return s.where(s.isin(keep), "Other")

def prepare(d):
    d = d.copy()
    d["floor_missing"] = d.floor.isna().astype(int)
    d["floor"] = d.floor.fillna(2.0)
    d["floor_ratio"] = (d.floor.clip(lower=0) / d.total_floors).clip(0, 1)
    d["bathroom_count"] = d.bathroom_count.fillna(1).clip(upper=6)
    d["has_maint_fee"] = (d.maintenance_fee.fillna(0) > 1).astype(int)
    o = d.orientation.fillna("")
    d["faces_south"] = o.str.contains("South").astype(int)
    d["n_directions"] = o.apply(lambda s: 0 if s == "" else len(s.split(",")))
    d["log_net_sqm"] = np.log(d.net_sqm)
    d["heating"] = lump(d.heating_type, ["Combi Boiler","Central (Metered)","Underfloor Heating","Central"])
    d["deed"] = lump(d.deed_status, ["Condominium Title","Construction Easement","Foreign Owner","Land Title","No Title Deed","Shared Title"])
    d["condition"] = d.building_condition.fillna("Missing")
    d["furnished"] = d.furnished.fillna("Missing")
    d["usage"] = lump(d.usage_status.fillna("Missing"), ["Vacant","Tenant-occupied","Owner-occupied"])
    d["credit"] = (d.credit_eligible == "Eligible").astype(int)
    d["exchange_yes"] = (d.exchange == "Yes").astype(int)
    return d

df = prepare(df)
df["log_price"] = np.log(df.price)

NUM = ["log_net_sqm","gross_sqm","rooms","halls","floor","floor_missing","floor_ratio","total_floors",
       "building_age","bathroom_count","is_in_complex","has_maint_fee","faces_south","n_directions",
       "credit","exchange_yes"]
CAT = ["district","heating","deed","condition","furnished","usage"]
print(len(NUM), "numeric +", len(CAT), "categorical feature groups =", len(NUM)+len(CAT))
""")
code(r"""
fig, ax = plt.subplots(figsize=(9, 7.5))
corr = df[NUM + ["log_price"]].corr()
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax,
            cbar_kws={"shrink": .7})
ax.set_title("Correlation matrix (numeric features)")
save("03_corr")
""")
code(r"""
order = df.groupby("district").log_price.median().sort_values().index
fig, ax = plt.subplots(figsize=(12, 4.5))
sns.boxplot(data=df, x="district", y="log_price", order=order, color="#bfdbfe", fliersize=1, ax=ax)
ax.tick_params(axis="x", rotation=75); ax.set_title("log(price) by district — location matters a lot"); ax.set_xlabel("")
save("04_district_box")
""")
code(r"""
fig, ax = plt.subplots(figsize=(6.5, 4.5))
smp = df.sample(4000, random_state=RNG)
ax.scatter(smp.log_net_sqm, smp.log_price, s=5, alpha=.35, color=ACC)
ax.set_xlabel("log(net_sqm)"); ax.set_ylabel("log(price)"); ax.set_title("log(price) vs log(net area)")
save("05_scatter_area")
""")
code(r"""
X_all = pd.get_dummies(df[NUM + CAT], columns=CAT, drop_first=True, dtype=float)
y_all = df.log_price.values
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=RNG)
print("train", X_tr.shape, " test", X_te.shape)
GROUPS = {f: [c for c in X_all.columns if c == f or c.startswith(f + "_")] for f in NUM + CAT}
GROUPS = {k: [c for c in v if (k in CAT) or c == k] for k, v in GROUPS.items()}
""")

md("## 4. Modeling（建模）")
md(r"""
### 4.1 基準模型：簡單線性回歸 log(price) ~ log(net_sqm)
""")
code(r"""
def metrics(y, yhat, p):
    n = len(y); r2 = r2_score(y, yhat)
    return {"R2": r2, "adjR2": 1-(1-r2)*(n-1)/(n-p-1),
            "RMSE_log": np.sqrt(mean_squared_error(y, yhat)),
            "MAE_log": mean_absolute_error(y, yhat),
            "MAPE_price_%": 100*np.mean(np.abs(np.exp(yhat)-np.exp(y))/np.exp(y)),
            "MedAPE_price_%": 100*np.median(np.abs(np.exp(yhat)-np.exp(y))/np.exp(y))}

def fit_ols(cols):
    return sm.OLS(y_tr, sm.add_constant(X_tr[cols])).fit()

results = {}
slr = fit_ols(["log_net_sqm"])
results["SLR (log_net_sqm)"] = metrics(y_te, slr.predict(sm.add_constant(X_te[["log_net_sqm"]])), 1)
print(slr.summary().tables[1])
""")
md("### 4.2 全特徵多元線性回歸（Full MLR）")
code(r"""
full_cols = list(X_all.columns)
full = fit_ols(full_cols)
results["MLR full"] = metrics(y_te, full.predict(sm.add_constant(X_te[full_cols])), len(full_cols))
print(f"features: {len(full_cols)}   train adjR2={full.rsquared_adj:.4f}   AIC={full.aic:.0f}")
""")
md(r"""
### 4.3 特徵選擇 (Feature Selection)

使用三種方法互相驗證：
1. **Forward Stepwise（以 AIC 為準則，特徵群組層級）**：每次加入能使 AIC 下降最多的特徵群組（類別變數的所有 dummy 視為同一群組），直到 AIC 無法再下降。
2. **RFE（Recursive Feature Elimination）**：以標準化後的線性回歸反覆刪除係數絕對值最小的特徵。
3. **LassoCV（L1 正則化）**：以 5-fold CV 選 α，係數被壓成 0 的特徵視為不重要。
""")
code(r"""
def group_cols(gs): return [c for g in gs for c in GROUPS[g]]

selected, remaining, history = [], list(GROUPS), []
best_aic = sm.OLS(y_tr, np.ones(len(y_tr))).fit().aic
while remaining:
    trials = [(fit_ols(group_cols(selected + [g])).aic, g) for g in remaining]
    aic, g = min(trials)
    if aic >= best_aic - 2:      # require meaningful improvement (ΔAIC > 2)
        break
    best_aic = aic; selected.append(g); remaining.remove(g)
    m = fit_ols(group_cols(selected))
    history.append({"step": len(selected), "added": g, "AIC": aic, "adjR2": m.rsquared_adj, "n_cols": len(group_cols(selected))})
hist = pd.DataFrame(history)
print("dropped groups:", remaining)
hist
""")
code(r"""
fig, ax1 = plt.subplots(figsize=(10, 4.2))
ax1.plot(hist.step, hist.adjR2, "o-", color=ACC); ax1.set_ylabel("adjusted R² (train)", color=ACC)
ax1.set_xticks(hist.step); ax1.set_xticklabels(hist.added, rotation=60, ha="right")
ax2 = ax1.twinx(); ax2.plot(hist.step, hist.AIC, "s--", color=ACC2); ax2.set_ylabel("AIC", color=ACC2); ax2.grid(False)
ax1.set_title("Forward stepwise selection (AIC criterion)")
save("06_stepwise")
""")
code(r"""
sc = StandardScaler().fit(X_tr)
Xs_tr = sc.transform(X_tr)
rfe = RFE(LinearRegression(), n_features_to_select=30).fit(Xs_tr, y_tr)
rfe_cols = list(X_tr.columns[rfe.support_])
lasso = LassoCV(cv=5, random_state=RNG, n_alphas=60).fit(Xs_tr, y_tr)
lasso_coef = pd.Series(lasso.coef_, index=X_tr.columns)
lasso_cols = list(lasso_coef[lasso_coef != 0].index)
print(f"RFE kept {len(rfe_cols)} | Lasso alpha={lasso.alpha_:.5f} kept {len(lasso_cols)}/{X_tr.shape[1]}")

# compare at the feature-group level
cmp = pd.DataFrame({g: {"Stepwise-AIC": g in selected,
                        "RFE(30)": any(c in rfe_cols for c in GROUPS[g]),
                        "Lasso": any(c in lasso_cols for c in GROUPS[g])} for g in GROUPS}).T
cmp["votes"] = cmp.sum(axis=1)
cmp.sort_values("votes", ascending=False)
""")
code(r"""
top = lasso_coef[lasso_coef.abs() > 0].abs().sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(top.index[::-1], lasso_coef[top.index][::-1], color=[ACC if v > 0 else ACC2 for v in lasso_coef[top.index][::-1]])
ax.set_title("Top-20 standardized Lasso coefficients"); ax.set_xlabel("coefficient (per 1 SD)")
save("07_lasso_coef")
""")
md("### 4.4 最終模型（Stepwise-AIC 選出之特徵）")
code(r"""
sel_cols = group_cols(selected)
final = fit_ols(sel_cols)
results["MLR stepwise (final)"] = metrics(y_te, final.predict(sm.add_constant(X_te[sel_cols])), len(sel_cols))
results["MLR RFE(30)"] = metrics(y_te, fit_ols(rfe_cols).predict(sm.add_constant(X_te[rfe_cols])), len(rfe_cols))
results["Lasso (CV)"] = metrics(y_te, lasso.predict(sc.transform(X_te)), len(lasso_cols))
print(final.summary().tables[0])
""")
code(r"""
coef = pd.DataFrame({"coef": final.params, "p": final.pvalues, "CI_low": final.conf_int()[0], "CI_high": final.conf_int()[1]})
coef["effect_%"] = (np.exp(coef.coef) - 1) * 100
coef.drop(index="const").reindex(coef.drop(index="const").coef.abs().sort_values(ascending=False).index).head(25).round(4)
""")
md(r"""
**係數解讀（log-log / log-linear）**
- `log_net_sqm` 係數 β：淨面積增加 1%，價格約增加 β%（彈性）。
- 二元或 dummy 變數：價格變化約為 (e^β − 1)×100%，例如某行政區的 dummy 代表相對基準區（Adalar）的價差。
- `building_age`：每增加 1 年屋齡的百分比折價。
""")

md("## 5. Evaluation（評估）")
code(r"""
res = pd.DataFrame(results).T
kf = KFold(5, shuffle=True, random_state=RNG)
cv = {"SLR (log_net_sqm)": ["log_net_sqm"], "MLR full": full_cols, "MLR stepwise (final)": sel_cols, "MLR RFE(30)": rfe_cols}
res["CV5_R2"] = np.nan
for k, cols in cv.items():
    res.loc[k, "CV5_R2"] = cross_val_score(LinearRegression(), X_all[cols], y_all, cv=kf, scoring="r2").mean()
res["n_features"] = [1, len(full_cols), len(sel_cols), len(rfe_cols), len(lasso_cols)]
res.round(4)
""")
code(r"""
X_te_c = sm.add_constant(X_te[sel_cols])
yhat_te = final.predict(X_te_c)
resid = y_te - yhat_te
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].scatter(yhat_te, resid, s=4, alpha=.3, color=ACC); axes[0].axhline(0, color="k", lw=1)
axes[0].set_xlabel("fitted log(price)"); axes[0].set_ylabel("residual"); axes[0].set_title("Residuals vs fitted")
sm.qqplot(resid, line="45", fit=True, ax=axes[1], markersize=2, alpha=.4); axes[1].set_title("Normal Q-Q")
axes[2].hist(resid, bins=70, color=GREY); axes[2].set_title("Residual distribution")
save("08_residuals")
""")
code(r"""
num_in = [c for c in sel_cols if c in NUM]
Xv = sm.add_constant(X_tr[num_in])
vif = pd.DataFrame({"feature": num_in, "VIF": [variance_inflation_factor(Xv.values, i+1) for i in range(len(num_in))]})
vif.sort_values("VIF", ascending=False).round(2)
""")
md(r"""
### 5.1 預測圖（含信賴區間 CI 與預測區間 PI）

- **信賴區間 (CI)**：平均房價 E[y|x] 的不確定性（模型係數估計誤差）。
- **預測區間 (PI)**：單一新房屋價格的不確定性（係數誤差 + 個別噪音 σ²），因此比 CI 寬得多。
""")
code(r"""
pred = final.get_prediction(X_te_c).summary_frame(alpha=0.05)
cover = np.mean((y_te >= pred.obs_ci_lower) & (y_te <= pred.obs_ci_upper))
print(f"95% PI empirical coverage on test set: {cover*100:.1f}%")

idx = np.random.default_rng(RNG).choice(len(y_te), 150, replace=False)
o = idx[np.argsort(pred["mean"].values[idx])]
x = np.arange(len(o))
fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(x, np.exp(pred.obs_ci_lower.values[o])/1e6, np.exp(pred.obs_ci_upper.values[o])/1e6, color=ACC, alpha=.15, label="95% prediction interval")
ax.fill_between(x, np.exp(pred.mean_ci_lower.values[o])/1e6, np.exp(pred.mean_ci_upper.values[o])/1e6, color=ACC, alpha=.45, label="95% confidence interval")
ax.plot(x, np.exp(pred["mean"].values[o])/1e6, color=ACC, lw=1.5, label="predicted")
ax.scatter(x, np.exp(y_te[o])/1e6, s=12, color=ACC2, zorder=3, label="actual")
ax.set_yscale("log"); ax.set_xlabel("150 random test listings (sorted by prediction)"); ax.set_ylabel("price (million TRY, log scale)")
ax.set_title(f"Test-set predictions with 95% CI / PI   (PI coverage = {cover*100:.1f}%)"); ax.legend(loc="upper left")
save("09_pred_interval_test")
""")
code(r"""
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(np.exp(y_te)/1e6, np.exp(yhat_te)/1e6, s=4, alpha=.3, color=ACC)
lim = [0.8, 300]; ax.plot(lim, lim, "k--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("actual price (M TRY)"); ax.set_ylabel("predicted price (M TRY)")
ax.set_title(f"Actual vs predicted (test R² = {r2_score(y_te, yhat_te):.3f})")
save("10_actual_vs_pred")
""")
code(r"""
# Effect plot: price vs net area for a typical apartment in three districts, with CI & PI bands
base = X_tr[sel_cols].median().to_frame().T
for c in sel_cols:
    if c.startswith("district_"): base[c] = 0.0   # other categorical dummies keep their mode (median of 0/1)
grid = np.linspace(40, 300, 60)
fig, ax = plt.subplots(figsize=(10, 5))
for dist, col in [("Esenyurt", ACC2), ("Kadıköy", ACC), ("Beşiktaş", "#16a34a")]:
    G = pd.concat([base]*len(grid), ignore_index=True)
    G["log_net_sqm"] = np.log(grid)
    if "gross_sqm" in G: G["gross_sqm"] = grid * 1.18
    if f"district_{dist}" in G: G[f"district_{dist}"] = 1.0
    sf = final.get_prediction(sm.add_constant(G, has_constant="add")).summary_frame(alpha=.05)
    ax.fill_between(grid, np.exp(sf.obs_ci_lower)/1e6, np.exp(sf.obs_ci_upper)/1e6, color=col, alpha=.10)
    ax.fill_between(grid, np.exp(sf.mean_ci_lower)/1e6, np.exp(sf.mean_ci_upper)/1e6, color=col, alpha=.45)
    ax.plot(grid, np.exp(sf["mean"])/1e6, color=col, lw=2, label=dist)
ax.set_xlabel("net area (m²)"); ax.set_ylabel("price (million TRY)")
ax.set_title("Predicted price vs net area — dark band: 95% CI, light band: 95% PI"); ax.legend()
save("11_effect_area_district")
""")
md(r"""
### 5.2 評估結論
- 多元線性回歸相較只用面積的簡單回歸，R² 大幅提升，主因是**行政區（地段）**提供了大量解釋力。
- Stepwise-AIC 刪除少數無貢獻的特徵群組後，測試表現與全模型相當但更精簡；RFE、Lasso 的結果與其高度一致，顯示選擇結果穩定。
- 殘差大致對稱，但尾部略厚（Q-Q 圖兩端偏離），代表仍有模型無法解釋的極端物件；log 轉換已明顯改善異質變異。
- 95% 預測區間的經驗覆蓋率接近 95%，區間可用於實務估價的「合理價格範圍」。
- **限制**：未使用 neighborhood（街區）等更細的地段資訊；房價可能存在非線性交互作用（例如面積 × 行政區），線性模型無法完全捕捉。
""")

md("## 6. Deployment（部署）")
code(r"""
# Export the final model for the GitHub Pages predictor (pure JS, no server)
cols = ["const"] + sel_cols
export = {
    "columns": cols,
    "coef": final.params[cols].tolist(),
    "cov": final.cov_params().loc[cols, cols].values.tolist(),
    "sigma2": float(final.scale),
    "df_resid": float(final.df_resid),
    "t975": float(__import__("scipy").stats.t.ppf(0.975, final.df_resid)),
    "num_defaults": {c: float(X_tr[c].median()) for c in sel_cols if c in NUM},
    "categories": {g: sorted(df[g].unique().tolist()) for g in CAT if g in selected},
    "base_levels": {g: sorted(df[g].unique().tolist())[0] for g in CAT if g in selected},
    "metrics": {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in res.T.to_dict().items()},
    "pi_coverage": float(cover), "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "selected_groups": selected, "dropped_groups": remaining,
}
with open("docs/model.json", "w", encoding="utf-8") as f: json.dump(export, f, ensure_ascii=False)
with open("docs/model.js", "w", encoding="utf-8") as f:   # same data as a script, so the page also works when opened locally
    f.write("window.MODEL = " + json.dumps(export, ensure_ascii=False) + ";")
res.round(4).to_csv("docs/assets/model_comparison.csv")
hist.round(4).to_csv("docs/assets/stepwise_history.csv", index=False)
coef.round(5).to_csv("docs/assets/final_coefficients.csv")
print("exported", len(cols), "coefficients")
""")
md(r"""
- 最終模型的係數、共變異數矩陣與殘差變異數已匯出成 `docs/model.json`。
- 網頁 `docs/index.html` 部署在 **GitHub Pages**，使用者輸入房屋條件後，瀏覽器端以 JavaScript 計算：
  - 點預測 ŷ = xᵀβ
  - 95% 預測區間 ŷ ± t₀.₉₇₅ · √(σ² + xᵀ Cov(β) x)，再以 e^(·) 轉回里拉
- 不需後端伺服器，可隨資料更新重新執行 notebook 後覆蓋 `model.json`。
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "HW2_MLR_CRISP-DM.ipynb")
print("notebook written with", len(cells), "cells")
