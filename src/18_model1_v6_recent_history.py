# step 54
# Next: Model 1 V6 — Recent-history training
# V6 start karte hain.
#
# Is baar hypothesis simple hai:
#
# Recent plant history may represent the 2017–2018 operating regime better than using the entire 2010–2016 history.
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


forecast_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_WeatherSolar_NextHour.csv"
)

preprocessed_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)


# ============================================================
# LOAD MODEL 1 DATA
# ============================================================

df = pd.read_csv(forecast_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


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


# ============================================================
# V6 PARAMETERS
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
# RECENT-HISTORY FOLDS
# ============================================================

folds = [
    ("Fold 1", 2012, 2013, 2014),
    ("Fold 2", 2013, 2014, 2015),
    ("Fold 3", 2014, 2015, 2016)
]


results = []


# ============================================================
# RUN V6 FOLDS
# ============================================================

print("=" * 70)
print("STEP 54 - MODEL 1 V6: RECENT-HISTORY TRAINING")
print("=" * 70)


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
    # MODEL PREDICTION
    # --------------------------------------------------------

    model_pred = model.predict(
        X_val,
        num_iteration=model.best_iteration_
    )


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    model_mae = mean_absolute_error(
        y_val,
        model_pred
    )

    model_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            model_pred
        )
    )

    model_r2 = r2_score(
        y_val,
        model_pred
    )


    print("\nModel 1 V6:")
    print("Best iteration:", model.best_iteration_)
    print("MAE :", round(model_mae, 4))
    print("RMSE:", round(model_rmse, 4))
    print("R²  :", round(model_r2, 4))


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({
        "fold": fold_name,
        "validation_year": val_year,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "mae": model_mae,
        "rmse": model_rmse,
        "r2": model_r2
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("MODEL 1 V6 ROLLING VALIDATION SUMMARY")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


print("\nAverage V6 performance:")

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


print("\nReference - Full-history V1 average:")

print("MAE :", 15.4730)
print("RMSE:", 30.2403)
print("R²  :", 0.9088)