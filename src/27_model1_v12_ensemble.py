from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from xgboost import XGBRegressor

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
df["target_timestamp"] = pd.to_datetime(
    df["target_timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


print("=" * 70)
print("STEP 63 - MODEL 1 V12: LIGHTGBM + XGBOOST ENSEMBLE")
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

print("\nFeature count:", len(features))


# ============================================================
# ROLLING FOLDS
# ============================================================

folds = [
    ("Fold 1", 2012, 2013, 2014),
    ("Fold 2", 2013, 2014, 2015),
    ("Fold 3", 2014, 2015, 2016)
]


# ============================================================
# MODEL PARAMETERS
# ============================================================

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1
}


xgb_params = {
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "early_stopping_rounds": 50,
    "n_jobs": -1
}


# ============================================================
# STORE OUT-OF-FOLD PREDICTIONS
# ============================================================

oof_actual = []
oof_lgb = []
oof_xgb = []


# ============================================================
# RESULTS
# ============================================================

fold_results = []


# ============================================================
# RUN FOLDS
# ============================================================

for (
        fold_name,
        train_start_year,
        train_end_year,
        val_year
) in folds:

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
    # SPLIT
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


    # ========================================================
    # LIGHTGBM
    # ========================================================

    lgb_model = lgb.LGBMRegressor(
        **lgb_params
    )

    lgb_model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )

    lgb_pred = lgb_model.predict(
        X_val,
        num_iteration=lgb_model.best_iteration_
    )


    # ========================================================
    # XGBOOST
    # ========================================================

    xgb_model = XGBRegressor(
        **xgb_params
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val)
        ],
        verbose=False
    )

    xgb_pred = xgb_model.predict(
        X_val
    )


    # ========================================================
    # STORE OOF PREDICTIONS
    # ========================================================

    oof_actual.extend(
        y_val.values
    )

    oof_lgb.extend(
        lgb_pred
    )

    oof_xgb.extend(
        xgb_pred
    )


    # ========================================================
    # INDIVIDUAL MODEL METRICS
    # ========================================================

    lgb_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            lgb_pred
        )
    )

    xgb_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            xgb_pred
        )
    )


    print("\nLightGBM RMSE:",
          round(lgb_rmse, 4))

    print("XGBoost RMSE:",
          round(xgb_rmse, 4))


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

oof_actual = np.array(oof_actual)

oof_lgb = np.array(oof_lgb)

oof_xgb = np.array(oof_xgb)


# ============================================================
# FIND GLOBAL ENSEMBLE WEIGHT
# ============================================================

print("\n" + "=" * 70)
print("ENSEMBLE WEIGHT SEARCH")
print("=" * 70)


best_weight = None
best_rmse = float("inf")


# weight = LightGBM weight
# XGBoost weight = 1 - weight

for weight in np.arange(
        0.0,
        1.01,
        0.05
):

    ensemble_pred = (
            weight * oof_lgb
            +
            (1 - weight) * oof_xgb
    )

    rmse = np.sqrt(
        mean_squared_error(
            oof_actual,
            ensemble_pred
        )
    )

    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = weight


best_xgb_weight = 1 - best_weight


print(
    "\nBest LightGBM weight:",
    round(best_weight, 2)
)

print(
    "Best XGBoost weight:",
    round(best_xgb_weight, 2)
)


# ============================================================
# FINAL OOF ENSEMBLE
# ============================================================

ensemble_oof_pred = (
        best_weight * oof_lgb
        +
        best_xgb_weight * oof_xgb
)


# ============================================================
# METRICS
# ============================================================

ensemble_mae = mean_absolute_error(
    oof_actual,
    ensemble_oof_pred
)

ensemble_rmse = np.sqrt(
    mean_squared_error(
        oof_actual,
        ensemble_oof_pred
    )
)

ensemble_r2 = r2_score(
    oof_actual,
    ensemble_oof_pred
)


# ============================================================
# INDIVIDUAL OOF METRICS
# ============================================================

lgb_mae = mean_absolute_error(
    oof_actual,
    oof_lgb
)

lgb_rmse = np.sqrt(
    mean_squared_error(
        oof_actual,
        oof_lgb
    )
)

lgb_r2 = r2_score(
    oof_actual,
    oof_lgb
)


xgb_mae = mean_absolute_error(
    oof_actual,
    oof_xgb
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        oof_actual,
        oof_xgb
    )
)

xgb_r2 = r2_score(
    oof_actual,
    oof_xgb
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 V12 ENSEMBLE RESULTS")
print("=" * 70)


print("\nLightGBM V6:")
print("MAE :", round(lgb_mae, 4))
print("RMSE:", round(lgb_rmse, 4))
print("R²  :", round(lgb_r2, 4))


print("\nXGBoost V8:")
print("MAE :", round(xgb_mae, 4))
print("RMSE:", round(xgb_rmse, 4))
print("R²  :", round(xgb_r2, 4))


print("\nV12 Ensemble:")
print("MAE :", round(ensemble_mae, 4))
print("RMSE:", round(ensemble_rmse, 4))
print("R²  :", round(ensemble_r2, 4))


print("\nReferences:")
print("V6 LightGBM RMSE:", 30.0998)
print("V8 XGBoost RMSE:", 29.9999)
print("Persistence RMSE:", 42.2669)