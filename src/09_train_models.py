# 🚀 Step 35 — Model 1 + Model 2 Training
from pathlib import Path
import pandas as pd
import lightgbm as lgb


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

model1_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_WeatherSolar_NextHour.csv"
)

model2_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model2_Intraday_NextHour.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

model1_df = pd.read_csv(model1_path)
model2_df = pd.read_csv(model2_path)

model1_df["timestamp"] = pd.to_datetime(model1_df["timestamp"])
model2_df["timestamp"] = pd.to_datetime(model2_df["timestamp"])


# print("=" * 70)
# print("STEP 35 - TRAIN MODEL 1 AND MODEL 2")
# print("=" * 70)
#
# print("\nModel 1 dataset:", model1_df.shape)
# print("Model 2 dataset:", model2_df.shape)


# ============================================================
# SAME CHRONOLOGICAL SPLIT
# ============================================================

train_end = pd.Timestamp("2015-12-31 23:59:59")
val_start = pd.Timestamp("2016-01-01")
val_end = pd.Timestamp("2016-12-31 23:59:59")
test_start = pd.Timestamp("2017-01-01")


# ============================================================
# MODEL 1 FEATURES
# ============================================================

model1_features = [
    "ambient_temp__5062",
    "module_temp__5063",
    "poa_irradiance__5061",
    "T2M",
    "RH2M",
    "WS10M",
    "WD10M",
    "PS",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
    "wind_gust_10m",
    "snowfall",
    "wind_speed_900_mb",
    "wind_direction_900_mb",
    "solar_zenith_angle",
    "solar_azimuth_angle",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos"
]

# ============================================================
# MODEL 2 FEATURES
# ============================================================

model2_features = [
                      "ac_power__5069"
                  ] + model1_features


target = "target_ac_power"


# ============================================================
# CREATE SPLITS
# ============================================================

def create_split(df, feature_cols):

    train = df[df["timestamp"] <= train_end]

    validation = df[
        (df["timestamp"] >= val_start) &
        (df["timestamp"] <= val_end)
        ]

    test = df[df["timestamp"] >= test_start]

    X_train = train[feature_cols]
    y_train = train[target]

    X_val = validation[feature_cols]
    y_val = validation[target]

    X_test = test[feature_cols]
    y_test = test[target]

    return X_train, y_train, X_val, y_val, X_test, y_test


model1_data = create_split(model1_df, model1_features)
model2_data = create_split(model2_df, model2_features)


X1_train, y1_train, X1_val, y1_val, X1_test, y1_test = model1_data
X2_train, y2_train, X2_val, y2_val, X2_test, y2_test = model2_data


# ============================================================
# VERIFY SPLITS
# ============================================================

# print("\nMODEL 1")
# print("Train:", X1_train.shape)
# print("Validation:", X1_val.shape)
# print("Test:", X1_test.shape)
#
# print("\nMODEL 2")
# print("Train:", X2_train.shape)
# print("Validation:", X2_val.shape)
# print("Test:", X2_test.shape)


# ============================================================
# LIGHTGBM PARAMETERS
# ============================================================

params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1
}


# ============================================================
# MODEL 1 TRAINING
# ============================================================

# print("\n" + "=" * 70)
# print("TRAINING MODEL 1 - WEATHER/SOLAR")
# print("=" * 70)

model1 = lgb.LGBMRegressor(**params)

model1.fit(
    X1_train,
    y1_train,
    eval_set=[(X1_val, y1_val)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ]
)

# print("\nModel 1 training completed.")
# print("Best iteration:", model1.best_iteration_)


# ============================================================
# MODEL 2 TRAINING
# ============================================================

# print("\n" + "=" * 70)
# print("TRAINING MODEL 2 - INTRADAY")
# print("=" * 70)

model2 = lgb.LGBMRegressor(**params)

model2.fit(
    X2_train,
    y2_train,
    eval_set=[(X2_val, y2_val)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ]
)

# print("\nModel 2 training completed.")
# print("Best iteration:", model2.best_iteration_)

# 🚀 Step 36 — Final Test Evaluation
# print("\n" + "=" * 70)
# print("STEP 36 - FINAL TEST SET PREDICTIONS")
# print("=" * 70)

# ------------------------------------------------------------
# MODEL 1 PREDICTIONS
# ------------------------------------------------------------

model1_test_pred = model1.predict(
    X1_test,
    num_iteration=model1.best_iteration_
)

# ------------------------------------------------------------
# MODEL 2 PREDICTIONS
# ------------------------------------------------------------

model2_test_pred = model2.predict(
    X2_test,
    num_iteration=model2.best_iteration_
)

# ------------------------------------------------------------
# PERSISTENCE BASELINE
# ------------------------------------------------------------

baseline_test_pred = X2_test["ac_power__5069"].values

# ------------------------------------------------------------
# VERIFY
# ------------------------------------------------------------

# print("\nPrediction shapes:")
#
# print("Model 1:", model1_test_pred.shape)
# print("Model 2:", model2_test_pred.shape)
# print("Baseline:", baseline_test_pred.shape)
#
# print("\nActual test target:", y1_test.shape)
#
# print("\nFirst 10 predictions:")
#
# print("\nModel 1:")
# print(model1_test_pred[:10])
#
# print("\nModel 2:")
# print(model2_test_pred[:10])
#
# print("\nBaseline:")
# print(baseline_test_pred[:10])
#
# print("\nActual:")
# print(y1_test.values[:10])


# Step 37 — Final Test Metrics
# print("\n" + "=" * 70)
# print("STEP 37 - FINAL TEST SET EVALUATION")
# print("=" * 70)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


# ============================================================
# EVALUATE ALL THREE
# ============================================================

baseline_mae, baseline_rmse, baseline_r2 = evaluate_model(
    y1_test,
    baseline_test_pred
)

model1_mae, model1_rmse, model1_r2 = evaluate_model(
    y1_test,
    model1_test_pred
)

model2_mae, model2_rmse, model2_r2 = evaluate_model(
    y1_test,
    model2_test_pred
)


# ============================================================
# RESULTS
# ============================================================

# print("\nFINAL TEST RESULTS")
#
# print("\nPersistence Baseline:")
# print("MAE :", round(baseline_mae, 4))
# print("RMSE:", round(baseline_rmse, 4))
# print("R²  :", round(baseline_r2, 4))
#
# print("\nModel 1 - Weather/Solar:")
# print("MAE :", round(model1_mae, 4))
# print("RMSE:", round(model1_rmse, 4))
# print("R²  :", round(model1_r2, 4))
#
# print("\nModel 2 - Intraday:")
# print("MAE :", round(model2_mae, 4))
# print("RMSE:", round(model2_rmse, 4))
# print("R²  :", round(model2_r2, 4))


# 🔥 Step 38 — Model 1 Error Analysis
# print("\n" + "=" * 70)
# print("STEP 38 - MODEL 1 ERROR ANALYSIS")
# print("=" * 70)

# Create test analysis dataframe
error_df = pd.DataFrame({
    "timestamp": model1_df.loc[
        model1_df["timestamp"] >= test_start, "timestamp"
    ].values,
    "actual": y1_test.values,
    "model1_pred": model1_test_pred,
    "baseline_pred": baseline_test_pred
})

# Absolute errors
error_df["model1_abs_error"] = (
        error_df["actual"] - error_df["model1_pred"]
).abs()

error_df["baseline_abs_error"] = (
        error_df["actual"] - error_df["baseline_pred"]
).abs()

# Difference:
# positive = Model 1 has larger error
# negative = Model 1 has smaller error
error_df["model1_vs_baseline_error"] = (
        error_df["model1_abs_error"]
        - error_df["baseline_abs_error"]
)

# print("\nMean Model 1 absolute error:",
#       round(error_df["model1_abs_error"].mean(), 4))
#
# print("Mean Baseline absolute error:",
#       round(error_df["baseline_abs_error"].mean(), 4))
#
# print("\nModel 1 better than baseline:",
#       (error_df["model1_vs_baseline_error"] < 0).sum())
#
# print("Model 1 worse than baseline:",
#       (error_df["model1_vs_baseline_error"] > 0).sum())
#
# print("Equal error:",
#       (error_df["model1_vs_baseline_error"] == 0).sum())
#
#
# # Largest Model 1 errors
# print("\nTop 10 Model 1 absolute errors:")
#
# print(
#     error_df[
#         [
#             "timestamp",
#             "actual",
#             "model1_pred",
#             "baseline_pred",
#             "model1_abs_error"
#         ]
#     ]
#     .sort_values("model1_abs_error", ascending=False)
#     .head(10)
# )


# Step 39 — Model 1 Error + Conditions Analysis
# print("\n" + "=" * 70)
# print("STEP 39 - MODEL 1 ERROR VS OPERATING CONDITIONS")
# print("=" * 70)


# Add important features from the test set
# Create test dataframe for Model 1
model1_test_condition = model1_df[
    model1_df["timestamp"] >= test_start
    ].copy()

test_condition_df = model1_test_condition[
    [
        "timestamp",
        "poa_irradiance__5061",
        "ambient_temp__5062",
        "module_temp__5063",
        "CLOUD_AMT",
        "solar_zenith_angle",
        "hour_sin",
        "hour_cos"
    ]
].copy()

test_condition_df["actual"] = y1_test.values
test_condition_df["model1_pred"] = model1_test_pred
test_condition_df["baseline_pred"] = baseline_test_pred

test_condition_df["model1_abs_error"] = (
        test_condition_df["actual"]
        - test_condition_df["model1_pred"]
).abs()


# ------------------------------------------------------------
# TOP 10 ERRORS WITH CONDITIONS
# ------------------------------------------------------------

top_errors = (
    test_condition_df
    .sort_values("model1_abs_error", ascending=False)
    .head(10)
)

# print("\nTop 10 Model 1 errors with conditions:")
#
# print(
#     top_errors[
#         [
#             "timestamp",
#             "actual",
#             "model1_pred",
#             "model1_abs_error",
#             "poa_irradiance__5061",
#             "ambient_temp__5062",
#             "module_temp__5063",
#             "CLOUD_AMT",
#             "solar_zenith_angle"
#         ]
#     ].to_string(index=False)
# )


# ------------------------------------------------------------
# ERROR BY IRRADIANCE LEVEL
# ------------------------------------------------------------

test_condition_df["irradiance_group"] = pd.cut(
    test_condition_df["poa_irradiance__5061"],
    bins=[-1, 100, 400, 800, 1200, float("inf")],
    labels=[
        "0-100",
        "100-400",
        "400-800",
        "800-1200",
        "1200+"
    ]
)

irradiance_error = (
    test_condition_df
    .groupby("irradiance_group", observed=False)
    .agg(
        records=("model1_abs_error", "size"),
        mean_mae=("model1_abs_error", "mean")
    )
)

# print("\nModel 1 error by irradiance:")
# print(irradiance_error)


# ------------------------------------------------------------
# ERROR BY CLOUD COVER
# ------------------------------------------------------------

test_condition_df["cloud_group"] = pd.cut(
    test_condition_df["CLOUD_AMT"],
    bins=[-1, 20, 50, 80, 100],
    labels=[
        "0-20",
        "20-50",
        "50-80",
        "80-100"
    ]
)

cloud_error = (
    test_condition_df
    .groupby("cloud_group", observed=False)
    .agg(
        records=("model1_abs_error", "size"),
        mean_mae=("model1_abs_error", "mean")
    )
)

# print("\nModel 1 error by cloud cover:")
# print(cloud_error)

# Step 40 — Prediction Bias Analysis
print("\n" + "=" * 70)
print("STEP 40 - MODEL 1 PREDICTION BIAS ANALYSIS")
print("=" * 70)

bias_df = test_condition_df.copy()

# Signed error
# Positive = model overpredicts
# Negative = model underpredicts
bias_df["signed_error"] = (
        bias_df["model1_pred"] - bias_df["actual"]
)

# ------------------------------------------------------------
# OVERALL BIAS
# ------------------------------------------------------------

print(
    "\nMean signed error:",
    round(bias_df["signed_error"].mean(), 4)
)


# ------------------------------------------------------------
# BIAS BY IRRADIANCE
# ------------------------------------------------------------

bias_by_irradiance = (
    bias_df
    .groupby("irradiance_group", observed=False)
    .agg(
        records=("signed_error", "size"),
        mean_signed_error=("signed_error", "mean"),
        mean_abs_error=("model1_abs_error", "mean")
    )
)

print("\nBias by irradiance:")
print(bias_by_irradiance)


# ------------------------------------------------------------
# BIAS BY CLOUD COVER
# ------------------------------------------------------------

bias_by_cloud = (
    bias_df
    .groupby("cloud_group", observed=False)
    .agg(
        records=("signed_error", "size"),
        mean_signed_error=("signed_error", "mean"),
        mean_abs_error=("model1_abs_error", "mean")
    )
)

print("\nBias by cloud cover:")
print(bias_by_cloud)


# ------------------------------------------------------------
# LARGE OVERPREDICTIONS
# ------------------------------------------------------------

print("\nTop 10 overpredictions:")

print(
    bias_df
    .sort_values("signed_error", ascending=False)
    [
        [
            "timestamp",
            "actual",
            "model1_pred",
            "signed_error",
            "poa_irradiance__5061",
            "CLOUD_AMT",
            "solar_zenith_angle"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# LARGE UNDERPREDICTIONS
# ------------------------------------------------------------

print("\nTop 10 underpredictions:")

print(
    bias_df
    .sort_values("signed_error", ascending=True)
    [
        [
            "timestamp",
            "actual",
            "model1_pred",
            "signed_error",
            "poa_irradiance__5061",
            "CLOUD_AMT",
            "solar_zenith_angle"
        ]
    ]
    .head(10)
    .to_string(index=False)
)