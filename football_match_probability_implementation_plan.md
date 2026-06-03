# Football Match Probability Prediction — Full Implementation Plan for AI Agent

> **Goal:** Xây dựng một notebook chạy end-to-end để dự đoán xác suất **home / draw / away** cho từng trận bóng đá, tối ưu theo **multi-class log-loss**, và xuất file submission đúng format của `output_example.csv`.

---

## 0. Executive Summary

Bài toán là **multi-class probability classification** với 3 nhãn: `home`, `draw`, `away`. Dữ liệu có cấu trúc kết hợp giữa:

- **Static match features:** thông tin trận hiện tại như `league_id`, `is_cup`, `home_team_coach_id`, `away_team_coach_id`, ngày thi đấu.
- **Historical sequence features:** 10 trận gần nhất của đội nhà và đội khách, gồm goal, opponent_goal, rating, opponent_rating, coach, league, is_cup, is_play_home, match_date.
- **Target:** `target` trong train, gồm `home`, `draw`, `away`.
- **Output:** xác suất 3 lớp, tổng mỗi dòng phải bằng **1.0**.

Chiến lược tốt nhất nên đi theo hướng **strong tabular baseline trước, deep sequence model sau, ensemble cuối cùng**:

1. **LightGBM / CatBoost tabular model** với feature engineering mạnh từ chuỗi 10 trận.
2. **GRU/LSTM sequence model** để học pattern tuần tự từ 10 trận gần nhất.
3. **OOF ensemble + weight optimization** để phối xác suất giữa các model sao cho log-loss validation thấp nhất.
4. **Probability clipping + row-wise normalization** trước khi tạo submission.

---

## 1. Dataset Contract

### 1.1 Input Files

Notebook phải đọc 3 file:

```text
train.csv
test.csv
output_example.csv
```

### 1.2 Observed Schema From Provided Files

Từ dữ liệu đính kèm:

```text
train.csv:          144,290 rows x 191 columns
test.csv:            72,711 rows x 189 columns
output_example.csv:  72,711 rows x 4 columns
```

`train.csv` có thêm 2 cột không có trong test:

```text
target
score
```

Trong đó:

- `target` là nhãn cần học.
- `score` là kết quả tỷ số thật của trận, chỉ xuất hiện ở train và phải **loại khỏi feature** vì sẽ gây leakage.

### 1.3 Required Output Format

Submission cuối cùng phải đúng format:

```text
id,home,draw,away
17761448,0.333,0.333,0.333
...
```

Yêu cầu bắt buộc:

- Giữ nguyên thứ tự `id` như trong `output_example.csv` hoặc `test.csv`.
- Có đúng 4 cột: `id`, `home`, `draw`, `away`.
- Mỗi dòng: `home + draw + away = 1.0`.
- Mỗi xác suất nằm trong `[0, 1]`.
- Không chứa NaN, inf, hoặc giá trị âm.

---

## 2. Core Design Principles

### 2.1 Optimize Probability, Not Accuracy

Metric chính là **multi-class log-loss**, nên model cần dự đoán xác suất tốt, không chỉ nhãn đúng/sai.

Notebook phải dùng:

```python
sklearn.metrics.log_loss
```

với thứ tự class cố định:

```python
CLASS_ORDER = ["home", "draw", "away"]
```

### 2.2 Prevent Leakage

Không được dùng các cột sau làm feature:

```text
target
score
```

Không tạo feature từ dữ liệu tương lai so với `match_date`.

Các cột lịch sử đã được chuẩn bị sẵn theo 10 trận gần nhất, nhưng vẫn cần kiểm tra logic ngày:

```text
home_team_history_match_date_i <= match_date
away_team_history_match_date_i <= match_date
```

Nếu phát hiện lịch sử có ngày sau `match_date`, không dùng trực tiếp cột đó hoặc tạo flag kiểm tra lỗi.

### 2.3 Build Strong Baseline Before Deep Learning

Vì dữ liệu hiện tại là tabular-heavy với lịch sử đã flatten thành 180 cột, **LightGBM với feature engineering tốt thường là baseline rất mạnh**.

Deep Learning nên được dùng như model bổ sung trong ensemble, không nên thay thế hoàn toàn Gradient Boosting.

### 2.4 OOF Prediction Is Mandatory

Notebook phải tạo **Out-of-Fold predictions** cho từng model để:

- Đánh giá log-loss trung thực hơn.
- Tối ưu weight ensemble.
- Tránh chọn weight dựa trên test.

---

## 3. Notebook Structure Required

AI agent phải tạo notebook theo các cell/section sau.

---

# Part A — Setup & Imports

## A1. Install Required Packages

Notebook nên hỗ trợ chạy trong local IDE hoặc Colab.

Required:

```python
pandas
numpy
scikit-learn
lightgbm
optuna
matplotlib
```

Optional nhưng khuyến nghị:

```python
catboost
tensorflow
torch
```

Nếu môi trường không có GPU hoặc thiếu TensorFlow/PyTorch, notebook vẫn phải chạy được bằng LightGBM-only mode.

## A2. Global Config

Tạo config rõ ràng:

```python
DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
OUTPUT_EXAMPLE_PATH = DATA_DIR / "output_example.csv"
SUBMISSION_PATH = DATA_DIR / "submission.csv"

RANDOM_STATE = 42
N_SPLITS = 5
CLASS_ORDER = ["home", "draw", "away"]
EPS = 1e-15
RUN_DEEP_MODEL = True
RUN_OPTUNA = False
```

`RUN_OPTUNA` nên để `False` mặc định để notebook chạy nhanh; có thể bật khi cần tuning sâu.

---

# Part B — Load Data & Basic Validation

## B1. Load CSV

Yêu cầu:

```python
train = pd.read_csv(TRAIN_PATH, low_memory=False)
test = pd.read_csv(TEST_PATH, low_memory=False)
output_example = pd.read_csv(OUTPUT_EXAMPLE_PATH)
```

## B2. Validate Required Columns

Kiểm tra:

- `id` tồn tại trong train/test/output_example.
- `target` tồn tại trong train.
- `target` chỉ gồm `home`, `draw`, `away`.
- `output_example` có đúng cột `id`, `home`, `draw`, `away`.

## B3. Print Overview

Notebook phải in:

```text
train shape
test shape
output shape
target distribution
missing rate top columns
date range train/test
```

## B4. Sort by Date

Convert:

```python
train["match_date"] = pd.to_datetime(train["match_date"], errors="coerce")
test["match_date"] = pd.to_datetime(test["match_date"], errors="coerce")
```

Sau đó sort train theo `match_date` để hỗ trợ time-based validation.

---

# Part C — Target Encoding

Map target:

```python
target_map = {"home": 0, "draw": 1, "away": 2}
inverse_target_map = {0: "home", 1: "draw", 2: "away"}
y = train["target"].map(target_map).astype(int)
```

Bắt buộc kiểm tra:

```python
assert y.notna().all()
```

---

# Part D — Feature Engineering

Đây là phần quan trọng nhất của notebook.

## D1. Column Groups

Tự động nhận diện group cột:

```python
static_cols = [
    "league_id",
    "is_cup",
    "home_team_coach_id",
    "away_team_coach_id",
    "match_date"
]

home_history_cols = columns starting with "home_team_history_"
away_history_cols = columns starting with "away_team_history_"
```

History attributes quan sát được:

```text
match_date
is_play_home
is_cup
goal
opponent_goal
rating
opponent_rating
coach
league_id
```

Với mỗi team side:

```text
home_team_history_<attribute>_<i>
away_team_history_<attribute>_<i>
```

Trong đó `i = 1..10`.

---

## D2. Basic Date Features

Tạo từ `match_date`:

```text
match_year
match_month
match_day
match_dayofweek
match_weekofyear
match_is_weekend
```

Có thể thêm cyclic encoding:

```text
month_sin
month_cos
dow_sin
dow_cos
```

---

## D3. Historical Time Gap Features

Với mỗi side `home`, `away`, convert:

```text
{side}_team_history_match_date_1..10
```

thành datetime.

Tạo:

```text
{side}_days_since_last_match_1
{side}_days_since_last_match_2
...
{side}_days_since_last_match_10
{side}_avg_rest_days_3
{side}_avg_rest_days_5
{side}_avg_rest_days_10
{side}_min_rest_days_10
{side}_max_rest_days_10
{side}_std_rest_days_10
```

Logic:

```python
days_since = (match_date - history_match_date_i).dt.days
```

Nếu `days_since < 0`, set thành NaN và tạo flag:

```text
{side}_has_future_history_flag
```

---

## D4. Historical Result Features

Với mỗi side, dùng:

```text
goal_i
opponent_goal_i
```

Tạo result từng trận:

```text
win_i  = 1 if goal_i > opponent_goal_i else 0
draw_i = 1 if goal_i == opponent_goal_i else 0
loss_i = 1 if goal_i < opponent_goal_i else 0
points_i = 3 * win_i + 1 * draw_i
goal_diff_i = goal_i - opponent_goal_i
```

Sau đó aggregate theo window:

```text
last_3
last_5
last_10
```

Feature cần có:

```text
{side}_points_sum_3
{side}_points_sum_5
{side}_points_sum_10

{side}_points_avg_3
{side}_points_avg_5
{side}_points_avg_10

{side}_win_rate_3
{side}_win_rate_5
{side}_win_rate_10

{side}_draw_rate_3
{side}_draw_rate_5
{side}_draw_rate_10

{side}_loss_rate_3
{side}_loss_rate_5
{side}_loss_rate_10

{side}_goals_avg_3
{side}_goals_avg_5
{side}_goals_avg_10

{side}_opp_goals_avg_3
{side}_opp_goals_avg_5
{side}_opp_goals_avg_10

{side}_goal_diff_avg_3
{side}_goal_diff_avg_5
{side}_goal_diff_avg_10

{side}_goal_diff_sum_10
{side}_goals_std_10
{side}_opp_goals_std_10
```

---

## D5. Recent Form Trend Features

Tạo feature thể hiện phong độ đang lên/xuống.

Ví dụ:

```text
{side}_points_trend_3_vs_10 = points_avg_3 - points_avg_10
{side}_goals_trend_3_vs_10 = goals_avg_3 - goals_avg_10
{side}_defense_trend_3_vs_10 = opp_goals_avg_3 - opp_goals_avg_10
{side}_rating_trend_3_vs_10 = rating_avg_3 - rating_avg_10
```

Ý nghĩa:

- `points_trend_3_vs_10 > 0`: phong độ gần đây tốt hơn trung bình 10 trận.
- `opp_goals_trend_3_vs_10 < 0`: phòng thủ gần đây tốt hơn.

---

## D6. Rating Strength Features

Với mỗi side:

```text
rating_i
opponent_rating_i
rating_diff_i = rating_i - opponent_rating_i
```

Aggregate:

```text
{side}_rating_avg_3
{side}_rating_avg_5
{side}_rating_avg_10

{side}_opp_rating_avg_3
{side}_opp_rating_avg_5
{side}_opp_rating_avg_10

{side}_rating_diff_avg_3
{side}_rating_diff_avg_5
{side}_rating_diff_avg_10

{side}_rating_diff_std_10
{side}_rating_diff_min_10
{side}_rating_diff_max_10
```

---

## D7. Home/Away Context Features

Dựa trên:

```text
{side}_team_history_is_play_home_i
```

Tạo:

```text
{side}_played_home_rate_3
{side}_played_home_rate_5
{side}_played_home_rate_10
```

Tạo split performance:

```text
{side}_points_when_play_home_avg_10
{side}_points_when_play_away_avg_10
{side}_goals_when_play_home_avg_10
{side}_goals_when_play_away_avg_10
```

Với home team hiện tại, form sân nhà quan trọng hơn; với away team hiện tại, form sân khách quan trọng hơn.

Tạo thêm:

```text
home_relevant_venue_points_avg_10 = home_points_when_play_home_avg_10
away_relevant_venue_points_avg_10 = away_points_when_play_away_avg_10
venue_form_diff = home_relevant_venue_points_avg_10 - away_relevant_venue_points_avg_10
```

---

## D8. Cup / League Consistency Features

Dựa trên:

```text
is_cup
{side}_team_history_is_cup_i
league_id
{side}_team_history_league_id_i
```

Tạo:

```text
{side}_cup_match_rate_10
{side}_same_league_rate_10
{side}_league_switch_count_10
```

Logic:

```python
same_league_i = history_league_id_i == current league_id
```

---

## D9. Coach Stability Features

Dựa trên:

```text
home_team_coach_id
away_team_coach_id
{side}_team_history_coach_i
```

Tạo:

```text
{side}_same_coach_rate_10
{side}_coach_change_flag
{side}_coach_unique_count_10
```

Logic:

```python
same_coach_i = history_coach_i == current_coach_id
coach_change_flag = same_coach_rate_10 < 1
```

Vì coach id missing khá nhiều, cần tạo thêm:

```text
home_coach_missing
away_coach_missing
```

---

## D10. Relative Matchup Features

Tạo feature chênh lệch giữa home và away:

```text
points_avg_diff_3  = home_points_avg_3 - away_points_avg_3
points_avg_diff_5  = home_points_avg_5 - away_points_avg_5
points_avg_diff_10 = home_points_avg_10 - away_points_avg_10

goals_avg_diff_3
goals_avg_diff_5
goals_avg_diff_10

opp_goals_avg_diff_3
opp_goals_avg_diff_5
opp_goals_avg_diff_10

goal_diff_avg_diff_3
goal_diff_avg_diff_5
goal_diff_avg_diff_10

rating_avg_diff_3
rating_avg_diff_5
rating_avg_diff_10

rating_diff_avg_diff_10
rest_days_diff_1
venue_form_diff
coach_stability_diff
```

Đây là nhóm feature có khả năng rất mạnh vì mô hình cần so sánh hai đội trong cùng trận.

---

## D11. Missingness Features

Tạo flag missing cho các nhóm quan trọng:

```text
{side}_history_available_count_goal
{side}_history_available_count_rating
{side}_history_available_count_coach
{side}_missing_history_rate_goal
{side}_missing_history_rate_rating
```

Với LightGBM, giữ NaN trong numeric columns. Nhưng missing indicator vẫn hữu ích vì missingness có thể mang thông tin.

---

## D12. Categorical Features

Các categorical candidates:

```text
league_id
league_name
is_cup
home_team_name
away_team_name
home_team_coach_id
away_team_coach_id
history league ids
history coach ids
```

Khuyến nghị:

### For LightGBM

- Convert categorical columns sang `category`.
- Với ID numeric categorical như coach/league, không scale như numeric continuous.
- Có thể dùng native categorical của LightGBM.

### For CatBoost Optional

- CatBoost xử lý categorical tốt, nhất là high-cardinality id/name.
- Nếu dùng CatBoost, pass `cat_features`.

### For Neural Model

- Không đưa raw high-cardinality ID trực tiếp vào LSTM nếu chưa embedding.
- Baseline neural chỉ nên dùng numeric sequence features trước.
- Nếu muốn nâng cấp, dùng embedding cho league/coach.

---

# Part E — Preprocessing Pipelines

## E1. LightGBM Preprocessing

Yêu cầu:

- Không impute numeric NaN.
- Convert bool/object hợp lý.
- Drop raw date string columns sau khi tạo date/time-gap feature.
- Drop target/leakage columns.

Feature matrix:

```python
X_train_tabular
X_test_tabular
```

Label:

```python
y
```

Cần đảm bảo train/test có cùng columns:

```python
X_train_tabular, X_test_tabular = X_train_tabular.align(X_test_tabular, join="left", axis=1)
```

Sau align, test thiếu cột nào thì fill bằng NaN.

## E2. Sequence Model Preprocessing

Tạo tensor:

```text
X_home_seq: (n_samples, 10, n_features_per_timestep)
X_away_seq: (n_samples, 10, n_features_per_timestep)
X_seq:      (n_samples, 10, 2 * n_features_per_timestep + optional_diff_features)
```

Recommended timestep features:

```text
goal
opponent_goal
goal_diff
points
is_play_home
is_cup
rating
opponent_rating
rating_diff
days_since_match
same_league
same_coach
```

Quan trọng: giữ thứ tự từ xa đến gần hoặc gần đến xa một cách nhất quán.

Khuyến nghị:

```text
timestep 0 = oldest match
timestep 9 = most recent match
```

Vì cột `_1` thường là trận gần nhất, cần reverse thứ tự:

```python
ordered_steps = [10, 9, 8, ..., 1]
```

### Missing Handling for Sequence

Với GRU/LSTM:

- Numeric NaN phải được impute.
- Tốt nhất dùng median từ train.
- Thêm mask/missing indicators nếu có thời gian.

Recommended:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

Fit imputer/scaler trên train fold only để tránh leakage.

---

# Part F — Validation Strategy

## F1. Primary CV: StratifiedKFold

Dùng:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Mục tiêu:

- Giữ tỷ lệ `home/draw/away` cân bằng giữa folds.
- Tạo OOF predictions ổn định.

## F2. Secondary Validation: Time-based Holdout

Vì dữ liệu bóng đá là time-series-like, nên cần thêm sanity check:

```text
train period: older matches
validation period: latest 15-20% by match_date
```

Mục tiêu:

- Kiểm tra model có generalize theo thời gian không.
- Tránh CV quá lạc quan nếu shuffle làm rò rỉ distribution gần nhau.

Implementation:

```python
cutoff = train["match_date"].quantile(0.8)
train_idx = train["match_date"] < cutoff
valid_idx = train["match_date"] >= cutoff
```

Báo cáo cả:

```text
CV log-loss
time holdout log-loss
```

## F3. Group Leakage Check Optional

Nếu có nhiều trận gần nhau cùng đội, có thể thử thêm GroupKFold theo `league_id` hoặc theo pair/team, nhưng không bắt buộc vì mục tiêu chính là dự đoán trận mới cùng hệ sinh thái league/team.

---

# Part G — Model 1: LightGBM Tabular

## G1. Baseline LightGBM Parameters

Start with:

```python
params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
```

Training:

```python
LGBMClassifier(**params)
```

Use callbacks:

```python
early_stopping(stopping_rounds=200)
log_evaluation(period=200)
```

## G2. OOF Training Loop

For each fold:

1. Split train/valid.
2. Fit model.
3. Predict validation probability.
4. Predict test probability.
5. Save model.
6. Average test predictions across folds.

Required outputs:

```python
oof_lgb = np.zeros((len(train), 3))
test_lgb = np.zeros((len(test), 3))
fold_scores = []
```

After CV:

```python
lgb_oof_logloss = log_loss(y, oof_lgb, labels=[0, 1, 2])
```

## G3. Feature Importance

After training:

- Aggregate feature importance across folds.
- Print top 50 features.
- Save optional CSV:

```text
feature_importance_lgb.csv
```

This helps debug whether engineered features are useful.

---

# Part H — Model 2: CatBoost Optional But Recommended

CatBoost is recommended as a second tabular model because:

- It can handle categorical variables naturally.
- It often complements LightGBM well in probability ensemble.

Use if package is available.

Suggested params:

```python
CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5,
    random_seed=RANDOM_STATE,
    verbose=200,
    early_stopping_rounds=200,
    allow_writing_files=False
)
```

OOF process same as LightGBM.

Outputs:

```python
oof_cat
test_cat
cat_oof_logloss
```

If CatBoost unavailable, skip gracefully.

---

# Part I — Model 3: GRU/LSTM Sequence Model

## I1. Architecture Recommendation

Use a compact GRU model first. GRU is usually faster and often competitive with LSTM.

Input:

```text
X_seq: shape (samples, 10, sequence_features)
X_static_small: optional numeric static features
```

Recommended architecture:

```text
Input sequence
→ Masking or missing indicator
→ Bidirectional GRU(64, return_sequences=True)
→ Dropout(0.2)
→ GRU(32)
→ Dense(64, relu)
→ Dropout(0.2)
→ Dense(3, softmax)
```

If adding static features:

```text
sequence branch + static branch → concatenate → dense → softmax
```

## I2. Loss & Metrics

Compile:

```python
loss="sparse_categorical_crossentropy"
optimizer=Adam(learning_rate=1e-3)
metrics=[]
```

Validation metric:

```python
sklearn.metrics.log_loss
```

Callbacks:

```python
EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5)
```

## I3. Fold Training

Use same folds as tabular model for comparable OOF.

For each fold:

- Fit imputer/scaler only on train fold sequence.
- Transform valid/test.
- Train neural model.
- Predict valid/test.
- Store OOF/test probabilities.

Outputs:

```python
oof_gru
test_gru
gru_oof_logloss
```

## I4. Fallback Rule

If neural model is unstable or worse than LightGBM by large margin:

```text
Do not force it into ensemble.
```

Only include model if it improves OOF ensemble log-loss.

---

# Part J — Hyperparameter Optimization With Optuna

## J1. Tuning Priority

Tune in this order:

1. LightGBM.
2. CatBoost if used.
3. GRU/LSTM only after tabular models are stable.

## J2. LightGBM Search Space

Use Optuna with 30-80 trials depending on compute.

Suggested search space:

```python
learning_rate: loguniform(0.005, 0.08)
num_leaves: int(31, 255)
max_depth: int(4, 12) or -1
min_child_samples: int(20, 200)
subsample: float(0.6, 1.0)
colsample_bytree: float(0.6, 1.0)
reg_alpha: loguniform(1e-4, 10)
reg_lambda: loguniform(1e-4, 20)
```

Objective:

```python
mean fold log_loss
```

Use early stopping inside each fold.

## J3. Practical Tuning Rule

For notebook stability:

- Default should train strong baseline without Optuna.
- Add an optional `RUN_OPTUNA=True` block.
- If Optuna finds better params, overwrite LightGBM params.
- Save best params to JSON:

```text
best_lgb_params.json
```

---

# Part K — Ensemble Strategy

## K1. Candidate Prediction Matrices

Available predictions:

```python
oof_lgb, test_lgb
oof_cat, test_cat
oof_gru, test_gru
```

## K2. Simple Average Baseline

Start with:

```python
ensemble = mean(predictions)
```

Compute OOF log-loss.

## K3. Optimize Ensemble Weights

Use scipy minimize or grid search.

Constraint:

```text
weights >= 0
sum(weights) = 1
```

Objective:

```python
log_loss(y, weighted_oof, labels=[0, 1, 2])
```

Pseudo:

```python
def weighted_pred(weights, preds):
    p = sum(w * pred for w, pred in zip(weights, preds))
    return normalize_probabilities(p)

best_weights = optimize(...)
```

Only include a model if it improves OOF log-loss.

## K4. Optional Class-wise Calibration

If OOF probabilities are poorly calibrated:

- Try temperature scaling or class-wise exponent adjustment.
- Keep only if OOF log-loss improves.

Simple calibration candidate:

```python
p_calibrated = p ** temperature
p_calibrated = p_calibrated / p_calibrated.sum(axis=1, keepdims=True)
```

Search `temperature` around:

```text
0.8, 0.9, 1.0, 1.1, 1.2
```

---

# Part L — Probability Post-processing

Before submission:

```python
def normalize_probabilities(p, eps=1e-15):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    return p
```

Check:

```python
assert np.all(np.isfinite(p))
assert np.all(p >= 0)
assert np.all(p <= 1)
assert np.allclose(p.sum(axis=1), 1.0)
```

---

# Part M — Submission Generation

Create:

```python
submission = pd.DataFrame({
    "id": test["id"].values,
    "home": final_test_pred[:, 0],
    "draw": final_test_pred[:, 1],
    "away": final_test_pred[:, 2],
})
```

Align to output example:

```python
submission = output_example[["id"]].merge(submission, on="id", how="left")
```

Validate:

```python
assert list(submission.columns) == ["id", "home", "draw", "away"]
assert submission.shape[0] == output_example.shape[0]
assert submission[["home", "draw", "away"]].notna().all().all()
assert np.allclose(submission[["home", "draw", "away"]].sum(axis=1), 1.0)
```

Save:

```python
submission.to_csv("submission.csv", index=False)
```

Print:

```python
submission.head()
submission[["home", "draw", "away"]].describe()
```

---

# Part N — Expected Notebook Outputs

Notebook phải output rõ:

```text
1. Data overview
2. Target distribution
3. Number of engineered features
4. LightGBM fold log-loss
5. LightGBM OOF log-loss
6. Optional CatBoost OOF log-loss
7. Optional GRU/LSTM OOF log-loss
8. Ensemble weights
9. Final ensemble OOF log-loss
10. Submission preview
11. Saved file path: submission.csv
```

---

# Part O — Robustness Requirements

AI agent phải đảm bảo notebook không crash nếu:

- TensorFlow/PyTorch không được cài.
- CatBoost không được cài.
- Một số cột categorical bị thiếu hoặc mixed dtype.
- `is_cup` có giá trị `True/False`, `0/1`, hoặc string.
- Một số history columns có NaN.
- Test thiếu một số feature sau engineering.

Implement helper functions:

```python
safe_to_datetime()
safe_bool_to_float()
safe_category()
normalize_probabilities()
validate_submission()
get_history_values()
make_team_history_features()
make_sequence_tensor()
```

---

# Part P — Detailed Helper Function Design

## P1. `safe_bool_to_float(series)`

Convert các kiểu:

```text
True, "True", "true", 1, "1" → 1.0
False, "False", "false", 0, "0" → 0.0
NaN → NaN
```

## P2. `make_team_history_features(df, side)`

Input:

```text
side in ["home", "away"]
```

Output:

Feature dataframe chứa toàn bộ aggregate cho side đó.

Steps:

1. Extract goal/opponent_goal/rating/opponent_rating/is_play_home/is_cup/league_id/coach/match_date.
2. Compute points, win/draw/loss, goal_diff, rating_diff.
3. Aggregate over windows 3/5/10.
4. Compute trends.
5. Compute rest-day features.
6. Compute same coach / same league.
7. Return dataframe indexed same as input.

## P3. `make_relative_features(features)`

Input: feature dataframe đã có home/away aggregate.

Output: diff features between home and away.

## P4. `build_tabular_features(train, test)`

Steps:

1. Concatenate train/test with marker `is_train`.
2. Generate features once to ensure same columns.
3. Drop leakage columns.
4. Split back.
5. Convert categorical columns.
6. Return `X_train`, `X_test`, `cat_cols`.

## P5. `build_sequence_tensor(df, fit_objects=None)`

Return:

```python
X_seq
fit_objects
```

For train fold:

- Fit imputer/scaler.

For validation/test:

- Reuse imputer/scaler from train fold.

---

# Part Q — Recommended Development Order for AI Agent

Implement notebook in this order:

1. Load data and validate output format.
2. Implement target mapping and basic cleaning.
3. Implement feature engineering for history aggregates.
4. Train LightGBM CV baseline.
5. Generate baseline submission.
6. Add feature importance.
7. Add optional Optuna.
8. Add optional CatBoost.
9. Add sequence tensor builder.
10. Add GRU/LSTM model.
11. Add OOF ensemble weight optimization.
12. Add final submission validation.

Do not start with deep learning before LightGBM baseline is working.

---

# Part R — Minimum Viable Strong Version

Nếu cần notebook chạy chắc chắn, version tối thiểu phải có:

```text
- LightGBM only
- Full historical aggregate features
- StratifiedKFold OOF
- Early stopping
- Probability normalization
- submission.csv
```

This should already be competitive and stable.

---

# Part S — Advanced Version

Nếu có thêm thời gian/compute:

```text
- CatBoost model
- GRU sequence model
- Optuna tuning
- Ensemble weight optimization
- Time-based holdout report
- Calibration search
```

---

# Part T — Important Modeling Notes

## T1. Draw Class Is Usually Hardest

Trong bóng đá, class `draw` thường khó dự đoán hơn `home/away`. Không nên dùng accuracy làm metric chính vì model có thể né class draw nhưng log-loss vẫn kém.

## T2. Do Not Overfit Team Names

`home_team_name` và `away_team_name` có thể hữu ích nhưng cũng có thể overfit. Với LightGBM/CatBoost, nên thử:

- Version A: include team names.
- Version B: exclude team names.

Chọn version có OOF log-loss tốt hơn.

## T3. Use Test Distribution Carefully

Không tune trực tiếp trên test. Chỉ dùng test để inspect missingness/schema.

## T4. Sequence Model May Not Always Win

Vì dữ liệu sequence đã được flatten và có nhiều NaN, GRU/LSTM chỉ nên được giữ nếu OOF ensemble tốt hơn.

## T5. Probability Clipping Matters

Log-loss phạt rất nặng nếu model dự đoán xác suất gần 0 cho class thật. Vì vậy final probabilities phải được clip nhẹ và normalize.

---

# Part U — Agent Prompt

Dùng prompt dưới đây để giao task cho AI coding agent.

```text
You are an expert machine learning engineer. Build a complete end-to-end Python notebook for football match probability prediction using the provided files: train.csv, test.csv, and output_example.csv.

Goal:
- Predict multi-class probabilities for target classes: home, draw, away.
- Optimize multi-class log-loss.
- Generate submission.csv with exact columns: id, home, draw, away.
- Ensure each probability row sums to 1.0.

Dataset facts:
- train.csv has columns target and score that are not present in test.csv.
- Use target as label.
- Never use score as a feature because it leaks the actual match result.
- test.csv must be predicted and aligned with output_example.csv.

Implementation requirements:
1. Load data with pandas and validate schema.
2. Convert target to numeric labels using class order: home=0, draw=1, away=2.
3. Create robust feature engineering from the 10 historical matches for home and away teams:
   - goals, opponent goals, goal difference
   - win/draw/loss flags
   - points: win=3, draw=1, loss=0
   - rolling aggregates over last 3, 5, and 10 matches
   - form trend features: last 3 average minus last 10 average
   - rating and opponent rating aggregates
   - rating difference aggregates
   - rest-day features from history match dates
   - same league rate
   - same coach rate and coach change flags
   - played-home / played-away rates
   - missingness flags
   - home-vs-away relative difference features
4. Build a strong LightGBM multiclass model with StratifiedKFold OOF validation and early stopping.
5. Report fold log-loss and full OOF log-loss.
6. Optionally add CatBoost if installed.
7. Optionally add a GRU/LSTM sequence model from the 10-match history tensor if TensorFlow or PyTorch is available.
8. Use OOF predictions to optimize ensemble weights. Include only models that improve OOF log-loss.
9. Clip and normalize final probabilities.
10. Create and validate submission.csv.

Notebook quality requirements:
- Must run from top to bottom.
- Must be robust to missing values and mixed boolean/object types.
- Must not crash if optional libraries are unavailable.
- Must print useful diagnostics at each major step.
- Must save final submission.csv.
- Include clear markdown headings and concise code comments.
```

---

# Part V — Reference Notes

Useful references for implementation decisions:

- LightGBM supports missing values by default and can use native categorical features.
- `sklearn.metrics.log_loss` expects probability predictions shaped `(n_samples, n_classes)`.
- TensorFlow/Keras masking is useful for sequence models when padded or missing timesteps should be skipped.
- Optuna can be used for efficient hyperparameter search and pruning.

Reference URLs:

```text
https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
https://www.tensorflow.org/guide/keras/understanding_masking_and_padding
https://optuna.readthedocs.io/
```

---

# Final Acceptance Criteria

Notebook is accepted only if:

```text
[ ] It loads train/test/output_example successfully.
[ ] It removes target and score from feature columns.
[ ] It builds engineered features from both home and away 10-match histories.
[ ] It trains at least one LightGBM multiclass model.
[ ] It reports OOF multi-class log-loss.
[ ] It generates final probabilities for test.
[ ] It validates that probabilities are finite, non-negative, <= 1, and row-sum to 1.
[ ] It writes submission.csv.
[ ] submission.csv has exactly the same id count as output_example.csv.
[ ] submission.csv has exactly columns: id, home, draw, away.
```
