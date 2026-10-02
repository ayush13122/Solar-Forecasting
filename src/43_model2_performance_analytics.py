from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# PROJECT PATH
# ============================================================
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = None
for parent in [CURRENT_DIR] + list(CURRENT_DIR.parents):
    if (parent / "data" / "processed").exists() and (parent / "src").exists():
        PROJECT_ROOT = parent
        break
if PROJECT_ROOT is None:
    raise FileNotFoundError("Project root could not be detected.")

print("=" * 70)
print("STEP 43 - MODEL 2 PERFORMANCE ANALYTICS")
print("=" * 70)
print(f"Project root: {PROJECT_ROOT}")

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "Model2_Intraday_NextHour.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "performance_analytics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
TARGET = "target_ac_power"
print(f"Input file : {INPUT_PATH}")
print(f"Output dir : {OUTPUT_DIR}")
print("\nDataset shape:", df.shape)

# ============================================================
# MODEL 2 FEATURES — exact Step 35 definition
# ============================================================
model1_features = [
    "ambient_temp__5062", "module_temp__5063", "poa_irradiance__5061",
    "T2M", "RH2M", "WS10M", "WD10M", "PS", "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "high_cloud_cover",
    "medium_cloud_cover", "low_cloud_cover", "wind_gust_10m", "snowfall",
    "wind_speed_900_mb", "wind_direction_900_mb", "solar_zenith_angle",
    "solar_azimuth_angle", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
]
model2_features = ["ac_power__5069"] + model1_features

# ============================================================
# SAME CHRONOLOGICAL SPLIT AS Step 35
# ============================================================
train_end = pd.Timestamp("2015-12-31 23:59:59")
val_start = pd.Timestamp("2016-01-01")
val_end = pd.Timestamp("2016-12-31 23:59:59")
test_start = pd.Timestamp("2017-01-01")

train = df[df["timestamp"] <= train_end].copy()
validation = df[(df["timestamp"] >= val_start) & (df["timestamp"] <= val_end)].copy()
test = df[df["timestamp"] >= test_start].copy()

X_train, y_train = train[model2_features], train[TARGET]
X_val, y_val = validation[model2_features], validation[TARGET]
X_test, y_test = test[model2_features], test[TARGET]

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)
print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)

# ============================================================
# EXACT LIGHTGBM PARAMETERS FROM Step 35
# ============================================================
params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1,
}

print("\n" + "=" * 70)
print("TRAINING MODEL 2")
print("=" * 70)
model2 = lgb.LGBMRegressor(**params)
model2.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)
best_iteration = model2.best_iteration_
print("Best iteration:", best_iteration)

# ============================================================
# PREDICTIONS
# ============================================================
model2_pred = model2.predict(X_test, num_iteration=best_iteration)
baseline_pred = X_test["ac_power__5069"].values

perf = test[[
    "timestamp", "target_timestamp", "ac_power__5069",
    "poa_irradiance__5061", "ambient_temp__5062", "module_temp__5063",
    "CLOUD_AMT", "solar_zenith_angle", "hour_sin", "hour_cos",
    "month_sin", "month_cos",
]].copy()
perf["actual"] = y_test.values
perf["model2_pred"] = model2_pred
perf["baseline_pred"] = baseline_pred
perf["model2_error"] = perf["model2_pred"] - perf["actual"]
perf["model2_abs_error"] = perf["model2_error"].abs()
perf["baseline_error"] = perf["baseline_pred"] - perf["actual"]
perf["baseline_abs_error"] = perf["baseline_error"].abs()
perf["model2_squared_error"] = perf["model2_error"] ** 2

# ============================================================
# OVERALL METRICS
# ============================================================
def metrics(actual, predicted):
    return (
        mean_absolute_error(actual, predicted),
        np.sqrt(mean_squared_error(actual, predicted)),
        r2_score(actual, predicted),
    )

model2_mae, model2_rmse, model2_r2 = metrics(perf["actual"], perf["model2_pred"])
baseline_mae, baseline_rmse, baseline_r2 = metrics(perf["actual"], perf["baseline_pred"])

metrics_df = pd.DataFrame({
    "model": ["Persistence Baseline", "Model 2 - Intraday LightGBM"],
    "MAE": [baseline_mae, model2_mae],
    "RMSE": [baseline_rmse, model2_rmse],
    "R2": [baseline_r2, model2_r2],
})
metrics_df.to_csv(OUTPUT_DIR / "model2_overall_metrics.csv", index=False)

print("\n" + "=" * 70)
print("OVERALL TEST PERFORMANCE")
print("=" * 70)
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nMAE change vs persistence : {(model2_mae-baseline_mae)/baseline_mae*100:.2f}%")
print(f"RMSE change vs persistence: {(model2_rmse-baseline_rmse)/baseline_rmse*100:.2f}%")

# ============================================================
# ERROR SUMMARY
# ============================================================
error_summary = pd.DataFrame({
    "metric": ["Mean Error", "Median Error", "Mean Absolute Error", "Error Std", "Minimum Error", "Maximum Error"],
    "value": [perf["model2_error"].mean(), perf["model2_error"].median(), perf["model2_abs_error"].mean(), perf["model2_error"].std(), perf["model2_error"].min(), perf["model2_error"].max()],
})
error_summary.to_csv(OUTPUT_DIR / "model2_error_summary.csv", index=False)
print("\nERROR DISTRIBUTION")
print(error_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# MODEL 2 VS PERSISTENCE
# ============================================================
perf["model2_better_than_baseline"] = perf["model2_abs_error"] < perf["baseline_abs_error"]
better_count = int(perf["model2_better_than_baseline"].sum())
worse_count = int((~perf["model2_better_than_baseline"]).sum())
comparison_df = pd.DataFrame({
    "category": ["Model 2 lower absolute error", "Model 2 higher/equal absolute error"],
    "records": [better_count, worse_count],
    "percentage": [better_count/len(perf)*100, worse_count/len(perf)*100],
})
comparison_df.to_csv(OUTPUT_DIR / "model2_vs_persistence_comparison.csv", index=False)
print("\nMODEL 2 VS PERSISTENCE")
print(comparison_df.to_string(index=False))

# ============================================================
# GROUP ANALYSIS HELPER
# ============================================================
def grouped_error(frame, group_col):
    return (frame.groupby(group_col, observed=False)
            .agg(records=("model2_abs_error", "size"),
                 MAE=("model2_abs_error", "mean"),
                 RMSE=("model2_squared_error", lambda x: np.sqrt(x.mean())),
                 mean_bias=("model2_error", "mean"),
                 actual_mean=("actual", "mean"),
                 predicted_mean=("model2_pred", "mean"))
            .reset_index())

# Hour
perf["hour"] = perf["timestamp"].dt.hour
hourly_error = grouped_error(perf, "hour")
hourly_error.to_csv(OUTPUT_DIR / "model2_hourly_performance.csv", index=False)

# Month
perf["month"] = perf["timestamp"].dt.month
monthly_error = grouped_error(perf, "month")
monthly_error.to_csv(OUTPUT_DIR / "model2_monthly_performance.csv", index=False)

# Year
perf["year"] = perf["timestamp"].dt.year
yearly_error = grouped_error(perf, "year")
yearly_error.to_csv(OUTPUT_DIR / "model2_yearly_performance.csv", index=False)

# Irradiance
perf["irradiance_group"] = pd.cut(
    perf["poa_irradiance__5061"],
    bins=[-np.inf, 100, 400, 800, 1200, np.inf],
    labels=["0-100", "100-400", "400-800", "800-1200", "1200+"],
)
irradiance_error = grouped_error(perf, "irradiance_group")
irradiance_error.to_csv(OUTPUT_DIR / "model2_irradiance_performance.csv", index=False)

# Daylight / night
perf["daylight"] = perf["poa_irradiance__5061"] > 20
daylight_summary = grouped_error(perf, "daylight")
daylight_summary["period"] = daylight_summary["daylight"].map({True: "Daylight", False: "Night/Low irradiance"})
daylight_summary = daylight_summary[["period", "records", "MAE", "RMSE", "mean_bias", "actual_mean", "predicted_mean"]]
daylight_summary.to_csv(OUTPUT_DIR / "model2_daylight_performance.csv", index=False)

# Cloud
perf["cloud_group"] = pd.cut(
    perf["CLOUD_AMT"], bins=[-np.inf, 20, 50, 80, 100], labels=["0-20", "20-50", "50-80", "80-100"]
)
cloud_error = grouped_error(perf, "cloud_group")
cloud_error.to_csv(OUTPUT_DIR / "model2_cloud_performance.csv", index=False)

print("\n" + "=" * 70)
print("HOURLY PERFORMANCE")
print("=" * 70)
print(hourly_error.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nMONTHLY PERFORMANCE")
print(monthly_error.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nIRRADIANCE PERFORMANCE")
print(irradiance_error.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nDAYLIGHT / NIGHT PERFORMANCE")
print(daylight_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nCLOUD PERFORMANCE")
print(cloud_error.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nYEARLY PERFORMANCE")
print(yearly_error.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# TOP 20 ERRORS
# ============================================================
top_errors = (perf[[
    "timestamp", "target_timestamp", "actual", "model2_pred", "model2_error",
    "model2_abs_error", "poa_irradiance__5061", "ambient_temp__5062",
    "module_temp__5063", "CLOUD_AMT", "solar_zenith_angle",
]].sort_values("model2_abs_error", ascending=False).head(20))
top_errors.to_csv(OUTPUT_DIR / "model2_top_20_errors.csv", index=False)
print("\nTOP 20 LARGEST ERRORS")
print(top_errors.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# PLOTS
# ============================================================
plt.figure(figsize=(8, 8))
plt.scatter(perf["actual"], perf["model2_pred"], s=12, alpha=0.35)
lo = min(perf["actual"].min(), perf["model2_pred"].min())
hi = max(perf["actual"].max(), perf["model2_pred"].max())
plt.plot([lo, hi], [lo, hi], linestyle="--")
plt.xlabel("Actual Next-Hour AC Power")
plt.ylabel("Predicted Next-Hour AC Power")
plt.title("Model 2 — Actual vs Predicted Power")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model2_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 6))
plt.hist(perf["model2_error"], bins=60, alpha=0.8)
plt.axvline(0, linestyle="--")
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Frequency")
plt.title("Model 2 — Residual Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model2_residual_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(hourly_error["hour"], hourly_error["MAE"], marker="o")
plt.xlabel("Hour of Day")
plt.ylabel("MAE")
plt.title("Model 2 — MAE by Hour of Day")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model2_mae_by_hour.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(monthly_error["month"], monthly_error["MAE"], marker="o")
plt.xlabel("Month")
plt.ylabel("MAE")
plt.title("Model 2 — MAE by Month")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model2_mae_by_month.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(irradiance_error["irradiance_group"].astype(str), irradiance_error["MAE"])
plt.xlabel("POA Irradiance Regime")
plt.ylabel("MAE")
plt.title("Model 2 — MAE by POA Irradiance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model2_mae_by_irradiance.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(perf["timestamp"], perf["model2_error"], linewidth=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Time")
plt.ylabel("Prediction Error")
plt.title("Model 2 — Residuals Over Time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model2_residuals_over_time.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# SAVE COMPLETE TEST PREDICTIONS + SUMMARY
# ============================================================
perf.to_csv(OUTPUT_DIR / "model2_performance_test_predictions.csv", index=False)

summary = pd.DataFrame({
    "item": [
        "Best iteration", "Test records", "Model 2 MAE", "Model 2 RMSE", "Model 2 R2",
        "Persistence MAE", "Persistence RMSE", "Persistence R2", "Mean residual",
        "Model 2 better than persistence", "Model 2 worse/equal than persistence",
    ],
    "value": [
        best_iteration, len(perf), model2_mae, model2_rmse, model2_r2,
        baseline_mae, baseline_rmse, baseline_r2, perf["model2_error"].mean(),
        better_count, worse_count,
    ],
})
summary.to_csv(OUTPUT_DIR / "model2_performance_summary.csv", index=False)

print("\n" + "=" * 70)
print("PERFORMANCE ANALYTICS COMPLETE")
print("=" * 70)
print(f"Best iteration : {best_iteration}")
print(f"Test records   : {len(perf)}")
print(f"Model 2 MAE    : {model2_mae:.4f}")
print(f"Model 2 RMSE   : {model2_rmse:.4f}")
print(f"Model 2 R2     : {model2_r2:.4f}")
print(f"Persistence MAE: {baseline_mae:.4f}")
print(f"Persistence RMSE: {baseline_rmse:.4f}")
print(f"Persistence R2 : {baseline_r2:.4f}")
print(f"Results saved in: {OUTPUT_DIR}")
print("Done.")
