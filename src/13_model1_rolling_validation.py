# Step 49 — Rolling / Expanding Validation
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_WeatherSolar_NextHour.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(input_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# MODEL 1 FEATURES
# ============================================================

features = [
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

target = "target_ac_power"


# ============================================================
# ROLLING / EXPANDING VALIDATION FOLDS
# ============================================================

folds = [
    ("Fold 1", 2013, 2014),
    ("Fold 2", 2014, 2015),
    ("Fold 3", 2015, 2016)
]


# ============================================================
# MODEL PARAMETERS
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
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# RUN EACH FOLD
# ============================================================

print("=" * 70)
print("STEP 49 - MODEL 1 ROLLING VALIDATION")
print("=" * 70)


for fold_name, train_end_year, val_year in folds:

    print("\n" + "-" * 70)
    print(f"{fold_name}: TRAIN 2010-{train_end_year} → VALIDATE {val_year}")
    print("-" * 70)

    train_start = pd.Timestamp("2010-01-01")
    train_end = pd.Timestamp(
        f"{train_end_year}-12-31 23:59:59"
    )

    val_start = pd.Timestamp(
        f"{val_year}-01-01"
    )

    val_end = pd.Timestamp(
        f"{val_year}-12-31 23:59:59"
    )

    # --------------------------------------------------------
    # TARGET-TIMESTAMP BASED SPLIT
    # --------------------------------------------------------

    train_df = df[
        (df["target_timestamp"] >= train_start) &
        (df["target_timestamp"] < val_start)
        ].copy()

    val_df = df[
        (df["target_timestamp"] >= val_start) &
        (df["target_timestamp"] <= val_end)
        ].copy()

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    val_pred = model.predict(
        X_val,
        num_iteration=model.best_iteration_
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_val,
        val_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            val_pred
        )
    )

    r2 = r2_score(
        y_val,
        val_pred
    )

    print("\nResults:")
    print("Best iteration:", model.best_iteration_)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    results.append({
        "fold": fold_name,
        "train_end_year": train_end_year,
        "validation_year": val_year,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("ROLLING VALIDATION SUMMARY")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


print("\nAverage validation performance:")

print(
    "MAE :",
    round(results_df["mae"].mean(), 4)
)

print(
    "RMSE:",
    round(results_df["rmse"].mean(), 4)
)

print(
    "R²  :",
    round(results_df["r2"].mean(), 4)
)