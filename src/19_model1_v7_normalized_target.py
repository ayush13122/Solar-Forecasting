#V7 — Target formulation / normalized power

# Isme dekhenge ki normalized AC power (AC / 449.28) predict karna raw kW predict karne se generalization improve karta hai ya nahi.
# Step 55 — Model 1 V7
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT ROOT
# ============================================================

current_path = Path(__file__).resolve()

PROJECT_ROOT = None

for parent in current_path.parents:
    if (parent / "data" / "processed").exists():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate project root containing data/processed."
    )


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


print("=" * 70)
print("STEP 55 - MODEL 1 V7: NORMALIZED TARGET")
print("=" * 70)

print("\nInput shape:", df.shape)


# ============================================================
# FEATURES
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

dc_capacity_kw = 449.28


print("\nFeature count:", len(features))


# ============================================================
# NORMALIZED TARGET
# ============================================================

df["target_power_normalized"] = (
        df[target] / dc_capacity_kw
)


print(
    "\nNormalized target range:",
    round(df["target_power_normalized"].min(), 6),
    "to",
    round(df["target_power_normalized"].max(), 6)
)


# ============================================================
# V6 SAME RECENT-HISTORY FOLDS
# ============================================================

folds = [
    ("Fold 1", 2012, 2013, 2014),
    ("Fold 2", 2013, 2014, 2015),
    ("Fold 3", 2014, 2015, 2016)
]


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


results = []


# ============================================================
# ROLLING VALIDATION
# ============================================================

for fold_name, train_start_year, train_end_year, val_year in folds:

    print("\n" + "-" * 70)
    print(
        f"{fold_name}: "
        f"TRAIN {train_start_year}-{train_end_year} "
        f"→ VALIDATE {val_year}"
    )
    print("-" * 70)


    train_start = pd.Timestamp(
        f"{train_start_year}-01-01"
    )

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
        (df["target_timestamp"] <= train_end)
        ].copy()

    val_df = df[
        (df["target_timestamp"] >= val_start) &
        (df["target_timestamp"] <= val_end)
        ].copy()


    X_train = train_df[features]
    y_train = train_df["target_power_normalized"]

    X_val = val_df[features]
    y_val = val_df["target_power_normalized"]


    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)


    # --------------------------------------------------------
    # TRAIN V7
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
    # PREDICT NORMALIZED POWER
    # --------------------------------------------------------

    val_pred_normalized = model.predict(
        X_val,
        num_iteration=model.best_iteration_
    )


    # Convert back to kW
    val_pred_kw = (
            val_pred_normalized
            * dc_capacity_kw
    )


    actual_kw = val_df[target].values


    # --------------------------------------------------------
    # METRICS IN kW
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual_kw,
        val_pred_kw
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_kw,
            val_pred_kw
        )
    )

    r2 = r2_score(
        actual_kw,
        val_pred_kw
    )


    print("\nV7 Results:")
    print("Best iteration:", model.best_iteration_)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))


    results.append({
        "fold": fold_name,
        "validation_year": val_year,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "best_iteration": model.best_iteration_
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("MODEL 1 V7 ROLLING VALIDATION SUMMARY")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


print("\nAverage V7 performance:")

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


print("\nReference - V6 average:")

print("MAE :", 14.7334)
print("RMSE:", 30.0998)
print("R²  :", 0.9097)