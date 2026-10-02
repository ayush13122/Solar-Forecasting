# Step 67 — V15 Two-Stage Forecasting
# ================================================================
# STEP 67 - MODEL 1 V15: TWO-STAGE FORECASTING
# ================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import pvlib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


print("=" * 70)
print("STEP 67 - MODEL 1 V15: TWO-STAGE FORECASTING")
print("=" * 70)


# ----------------------------------------------------------------
# 1. FIND PROJECT ROOT ROBUSTLY
# ----------------------------------------------------------------

current_path = Path(__file__).resolve()

PROJECT_ROOT = None

for parent in current_path.parents:
    if (parent / "data" / "processed").exists():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Project root not found. Expected a folder containing data/processed."
    )

print(f"\nProject root: {PROJECT_ROOT}")


# ----------------------------------------------------------------
# 2. FILE PATHS
# ----------------------------------------------------------------

FORECAST_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_WeatherSolar_NextHour.csv"
)

HISTORY_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)


# ----------------------------------------------------------------
# 3. SETTINGS
# ----------------------------------------------------------------

# Modeling threshold only.
# Target > 1 kW = Active generation.
ACTIVE_THRESHOLD = 1.0

CLASSIFICATION_THRESHOLD = 0.50


# ----------------------------------------------------------------
# 4. LOAD DATA
# ----------------------------------------------------------------

forecast_df = pd.read_csv(
    FORECAST_FILE,
    parse_dates=["timestamp", "target_timestamp"]
)

history_df = pd.read_csv(
    HISTORY_FILE,
    parse_dates=["timestamp"]
)

print(f"\nForecast dataset: {forecast_df.shape}")
print(f"History dataset : {history_df.shape}")


# ----------------------------------------------------------------
# 5. BASE FEATURES
# ----------------------------------------------------------------

base_features = [
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
    "day_of_year_cos",
]

target_col = "target_ac_power"


# ----------------------------------------------------------------
# 6. CREATE V14 PHYSICS-INFORMED FEATURES
# ----------------------------------------------------------------

print("\nCreating physics-informed features...")


# 6.1 Solar elevation
history_df["solar_elevation_deg"] = (
        90.0 - history_df["solar_zenith_angle"]
)


# 6.2 Sun cosine of zenith
zenith_rad = np.deg2rad(
    history_df["solar_zenith_angle"]
)

history_df["sun_cos_zenith"] = (
    np.cos(zenith_rad).clip(lower=0)
)


# 6.3 Daylight flag
history_df["daylight_flag"] = (
        history_df["solar_zenith_angle"] < 90
).astype(int)


# 6.4 Relative air mass
zenith_for_airmass = (
    history_df["solar_zenith_angle"]
    .clip(lower=0, upper=89.9)
)

relative_airmass = pvlib.atmosphere.get_relative_airmass(
    zenith_for_airmass,
    model="kastenyoung1989"
)

relative_airmass = pd.Series(
    relative_airmass,
    index=history_df.index
)

relative_airmass.loc[
    history_df["daylight_flag"] == 0
    ] = 0

history_df["relative_airmass"] = relative_airmass


# 6.5 Pressure-corrected air mass
history_df["pressure_corrected_airmass"] = (
        history_df["relative_airmass"]
        * history_df["PS"]
        / 1013.25
)


# 6.6 Module - ambient temperature
history_df["module_ambient_delta"] = (
        history_df["module_temp__5063"]
        - history_df["ambient_temp__5062"]
)


# 6.7 POA × module temperature
history_df["poa_module_temp_interaction"] = (
        history_df["poa_irradiance__5061"]
        * history_df["module_temp__5063"]
)


# 6.8 POA × ambient temperature
history_df["poa_ambient_temp_interaction"] = (
        history_df["poa_irradiance__5061"]
        * history_df["ambient_temp__5062"]
)


physics_features = [
    "solar_elevation_deg",
    "sun_cos_zenith",
    "daylight_flag",
    "relative_airmass",
    "pressure_corrected_airmass",
    "module_ambient_delta",
    "poa_module_temp_interaction",
    "poa_ambient_temp_interaction",
]


# ----------------------------------------------------------------
# 7. MERGE PHYSICS FEATURES
# ----------------------------------------------------------------

feature_source = history_df[
    ["timestamp"] + physics_features
    ].copy()

forecast_df = forecast_df.merge(
    feature_source,
    on="timestamp",
    how="left",
    validate="one_to_one"
)


# ----------------------------------------------------------------
# 8. FINAL FEATURES
# ----------------------------------------------------------------

v15_features = base_features + physics_features

print(f"\nBase features    : {len(base_features)}")
print(f"Physics features : {len(physics_features)}")
print(f"Total V15 features: {len(v15_features)}")


# ----------------------------------------------------------------
# 9. MISSING VALUE CHECK
# ----------------------------------------------------------------

missing_features = forecast_df[
    v15_features
].isna().sum()

total_missing = missing_features.sum()

print("\nMissing feature values:")
print(missing_features[missing_features > 0])

print(f"\nTotal missing feature values: {total_missing}")


# ----------------------------------------------------------------
# 10. CREATE STAGE-1 CLASSIFICATION TARGET
# ----------------------------------------------------------------

forecast_df["active_target"] = (
        forecast_df[target_col] > ACTIVE_THRESHOLD
).astype(int)


print("\nStage-1 target distribution:")

print(
    forecast_df["active_target"]
    .value_counts()
    .rename({
        0: "Near-zero",
        1: "Active"
    })
)

active_percentage = (
        forecast_df["active_target"].mean() * 100
)

print(
    f"\nActive target percentage: "
    f"{active_percentage:.2f}%"
)


# ----------------------------------------------------------------
# 11. EVALUATION FUNCTION
# ----------------------------------------------------------------

def regression_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ----------------------------------------------------------------
# 12. ROLLING VALIDATION
# ----------------------------------------------------------------

folds = [
    {
        "train_start": "2012-01-01",
        "train_end": "2013-12-31 23:59:59",
        "val_start": "2014-01-01",
        "val_end": "2014-12-31 23:59:59",
        "year": 2014,
    },
    {
        "train_start": "2013-01-01",
        "train_end": "2014-12-31 23:59:59",
        "val_start": "2015-01-01",
        "val_end": "2015-12-31 23:59:59",
        "year": 2015,
    },
    {
        "train_start": "2014-01-01",
        "train_end": "2015-12-31 23:59:59",
        "val_start": "2016-01-01",
        "val_end": "2016-12-31 23:59:59",
        "year": 2016,
    },
]


results = []


# ----------------------------------------------------------------
# 13. RUN EACH FOLD
# ----------------------------------------------------------------

for i, fold in enumerate(folds, start=1):

    print("\n" + "-" * 70)

    print(
        f"Fold {i}: TRAIN "
        f"{fold['train_start'][:4]}-"
        f"{fold['train_end'][:4]} "
        f"→ VALIDATE {fold['year']}"
    )

    print("-" * 70)


    # ------------------------------------------------------------
    # CREATE TRAIN / VALIDATION MASKS
    # ------------------------------------------------------------

    train_mask = (
            (forecast_df["timestamp"] >= fold["train_start"])
            & (forecast_df["timestamp"] <= fold["train_end"])
            & forecast_df[target_col].notna()
    )

    val_mask = (
            (forecast_df["timestamp"] >= fold["val_start"])
            & (forecast_df["timestamp"] <= fold["val_end"])
            & forecast_df[target_col].notna()
    )


    train_data = forecast_df.loc[
        train_mask
    ].copy()

    val_data = forecast_df.loc[
        val_mask
    ].copy()


    print(f"Total train rows: {len(train_data)}")
    print(f"Total val rows  : {len(val_data)}")


    # ============================================================
    # STAGE 1 - CLASSIFIER
    # ============================================================

    print("\nSTAGE 1 - ACTIVE / NEAR-ZERO CLASSIFIER")
    print("-" * 50)


    X_train_cls = train_data[
        v15_features
    ]

    y_train_cls = train_data[
        "active_target"
    ]

    X_val_cls = val_data[
        v15_features
    ]

    y_val_cls = val_data[
        "active_target"
    ]


    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )


    classifier.fit(
        X_train_cls,
        y_train_cls,
        eval_set=[
            (X_val_cls, y_val_cls)
        ],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=True
            )
        ],
    )


    # Probability of active generation
    val_probability = classifier.predict_proba(
        X_val_cls
    )[:, 1]


    # Convert probability → class
    val_active_prediction = (
            val_probability >= CLASSIFICATION_THRESHOLD
    ).astype(int)


    cls_accuracy = accuracy_score(
        y_val_cls,
        val_active_prediction
    )

    cls_precision = precision_score(
        y_val_cls,
        val_active_prediction,
        zero_division=0
    )

    cls_recall = recall_score(
        y_val_cls,
        val_active_prediction,
        zero_division=0
    )

    cls_f1 = f1_score(
        y_val_cls,
        val_active_prediction,
        zero_division=0
    )


    print(
        f"Best classifier iteration: "
        f"{classifier.best_iteration_}"
    )

    print(
        f"Accuracy : {cls_accuracy:.4f}"
    )

    print(
        f"Precision: {cls_precision:.4f}"
    )

    print(
        f"Recall   : {cls_recall:.4f}"
    )

    print(
        f"F1       : {cls_f1:.4f}"
    )


    # ============================================================
    # STAGE 2 - ACTIVE POWER REGRESSOR
    # ============================================================

    print("\nSTAGE 2 - ACTIVE POWER REGRESSOR")
    print("-" * 50)


    active_train = train_data[
        train_data["active_target"] == 1
        ].copy()

    active_val = val_data[
        val_data["active_target"] == 1
        ].copy()


    X_train_reg = active_train[
        v15_features
    ]

    y_train_reg = active_train[
        target_col
    ]

    X_val_reg = active_val[
        v15_features
    ]

    y_val_reg = active_val[
        target_col
    ]


    print(
        f"Active training rows   : "
        f"{len(active_train)}"
    )

    print(
        f"Active validation rows : "
        f"{len(active_val)}"
    )


    regressor = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )


    regressor.fit(
        X_train_reg,
        y_train_reg,
        eval_set=[
            (X_val_reg, y_val_reg)
        ],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=True
            )
        ],
    )


    # ------------------------------------------------------------
    # Evaluate Stage 2 independently
    # ------------------------------------------------------------

    active_reg_prediction = regressor.predict(
        X_val_reg
    )

    stage2_mae, stage2_rmse, stage2_r2 = (
        regression_metrics(
            y_val_reg,
            active_reg_prediction
        )
    )


    print(
        f"\nBest regressor iteration: "
        f"{regressor.best_iteration_}"
    )

    print(
        f"Active-only MAE : {stage2_mae:.4f}"
    )

    print(
        f"Active-only RMSE: {stage2_rmse:.4f}"
    )

    print(
        f"Active-only R²  : {stage2_r2:.4f}"
    )


    # ============================================================
    # FINAL TWO-STAGE PREDICTION
    # ============================================================

    # Stage 2 prediction for every validation row
    all_reg_prediction = regressor.predict(
        X_val_cls
    )


    # If classifier says inactive → predict zero.
    # If classifier says active → use Stage-2 regression output.

    final_prediction = np.where(
        val_active_prediction == 1,
        all_reg_prediction,
        0.0
    )


    # Prevent tiny negative regression outputs
    final_prediction = np.maximum(
        final_prediction,
        0.0
    )


    # ------------------------------------------------------------
    # Combined forecast metrics
    # ------------------------------------------------------------

    final_mae, final_rmse, final_r2 = (
        regression_metrics(
            val_data[target_col],
            final_prediction
        )
    )


    print("\nFINAL TWO-STAGE FORECAST:")
    print(
        f"MAE : {final_mae:.4f}"
    )

    print(
        f"RMSE: {final_rmse:.4f}"
    )

    print(
        f"R²  : {final_r2:.4f}"
    )


    results.append({

        "fold": i,

        "validation_year": fold["year"],

        "train_rows": len(train_data),

        "validation_rows": len(val_data),

        "active_train_rows": len(active_train),

        "active_validation_rows": len(active_val),

        "classifier_accuracy": cls_accuracy,

        "classifier_precision": cls_precision,

        "classifier_recall": cls_recall,

        "classifier_f1": cls_f1,

        "stage2_active_mae": stage2_mae,

        "stage2_active_rmse": stage2_rmse,

        "stage2_active_r2": stage2_r2,

        "final_mae": final_mae,

        "final_rmse": final_rmse,

        "final_r2": final_r2,

        "classifier_best_iteration":
            classifier.best_iteration_,

        "regressor_best_iteration":
            regressor.best_iteration_,
    })


# ----------------------------------------------------------------
# 14. SUMMARY
# ----------------------------------------------------------------

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("MODEL 1 V15 TWO-STAGE ROLLING VALIDATION SUMMARY")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


avg_final_mae = results_df[
    "final_mae"
].mean()

avg_final_rmse = results_df[
    "final_rmse"
].mean()

avg_final_r2 = results_df[
    "final_r2"
].mean()


avg_f1 = results_df[
    "classifier_f1"
].mean()


print("\nAverage V15 performance:")

print(
    f"Final MAE : {avg_final_mae:.4f}"
)

print(
    f"Final RMSE: {avg_final_rmse:.4f}"
)

print(
    f"Final R²  : {avg_final_r2:.4f}"
)

print(
    f"Stage-1 F1: {avg_f1:.4f}"
)


# ----------------------------------------------------------------
# 15. REFERENCE MODELS
# ----------------------------------------------------------------

print("\nReference - V10:")
print("MAE : 14.3502")
print("RMSE: 29.9448")
print("R²  : 0.9107")

print("\nReference - V13:")
print("MAE : 14.3788")
print("RMSE: 29.9159")
print("R²  : 0.9109")

print("\nReference - V14:")
print("MAE : 14.3198")
print("RMSE: 29.6485")
print("R²  : 0.9124")


# ----------------------------------------------------------------
# 16. SAVE RESULTS
# ----------------------------------------------------------------

OUTPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V15_Two_Stage_Validation.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nValidation results saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("V15 COMPLETE")
print("=" * 70)