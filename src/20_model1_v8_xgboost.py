# Step 56 — XGBoost Rolling Validation
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError(
        "XGBoost is not installed in the current virtual environment."
    )


# ============================================================
# FIND PROJECT ROOT
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
df["target_timestamp"] = pd.to_datetime(
    df["target_timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


print("=" * 70)
print("STEP 56 - MODEL 1 V8: XGBOOST")
print("=" * 70)

print("\nInput shape:", df.shape)


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

print("\nFeature count:", len(features))


# ============================================================
# RECENT-HISTORY FOLDS
# ============================================================

folds = [
    ("Fold 1", 2012, 2013, 2014),
    ("Fold 2", 2013, 2014, 2015),
    ("Fold 3", 2014, 2015, 2016)
]


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# RUN FOLDS
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
    # SPLIT USING TARGET TIMESTAMP
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
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]


    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)


    # --------------------------------------------------------
    # XGBOOST MODEL
    # --------------------------------------------------------

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=50,
        n_jobs=-1
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    val_pred = model.predict(
        X_val
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


    print("\nV8 Results:")
    print("Best iteration:", model.best_iteration)
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
        "best_iteration": model.best_iteration
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL 1 V8 XGBOOST - ROLLING VALIDATION SUMMARY")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


print("\nAverage V8 performance:")

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


print("\nReference - V6 LightGBM:")

print("MAE :", 14.7334)
print("RMSE:", 30.0998)
print("R²  :", 0.9097)


print("\nReference - Persistence:")

print("MAE :", 23.1324)
print("RMSE:", 42.2669)
print("R²  :", 0.8225)