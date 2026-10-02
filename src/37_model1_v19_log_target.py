# V19 — Log-transformed active power
# ================================================================
# STEP 73 - MODEL 1 V19: LOG-TRANSFORMED ACTIVE POWER
# ================================================================
#
# Based on V15 Two-Stage architecture.
#
# V19 change:
#   Stage 2 predicts log1p(Active AC Power)
#   Final prediction = expm1(prediction)
#
# Purpose:
#   Reduce sensitivity to the highly skewed active-power target
#   and test whether this improves generalization.
#
# Validation:
#   2012-2013 -> 2014
#   2013-2014 -> 2015
#   2014-2015 -> 2016
#
# 2017-2018 test is NOT used here.
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
print("STEP 73 - MODEL 1 V19: LOG-TRANSFORMED ACTIVE POWER")
print("=" * 70)


# ----------------------------------------------------------------
# 1. FIND ACTUAL PROJECT ROOT
# ----------------------------------------------------------------

current_path = Path(__file__).resolve()

PROJECT_ROOT = None

KNOWN_DATA_FILE = (
    "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)

for parent in current_path.parents:

    if (
            (parent / ".venv").exists()
            and
            (
                    parent
                    / "data"
                    / "processed"
                    / KNOWN_DATA_FILE
            ).exists()
    ):
        PROJECT_ROOT = parent
        break

    if (
            (parent / ".venv_tf").exists()
            and
            (
                    parent
                    / "data"
                    / "processed"
                    / KNOWN_DATA_FILE
            ).exists()
    ):
        PROJECT_ROOT = parent
        break


if PROJECT_ROOT is None:

    raise FileNotFoundError(
        "Actual project root not found."
    )


print(
    f"\nActual project root:\n{PROJECT_ROOT}"
)


# ----------------------------------------------------------------
# 2. PATHS
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
        / KNOWN_DATA_FILE
)

OUTPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V19_Log_Target_Validation.csv"
)


# ----------------------------------------------------------------
# 3. SETTINGS
# ----------------------------------------------------------------

ACTIVE_THRESHOLD = 1.0

CLASSIFICATION_THRESHOLD = 0.50


# ----------------------------------------------------------------
# 4. LOAD DATA
# ----------------------------------------------------------------

forecast_df = pd.read_csv(
    FORECAST_FILE,
    parse_dates=[
        "timestamp",
        "target_timestamp",
    ],
)

history_df = pd.read_csv(
    HISTORY_FILE,
    parse_dates=[
        "timestamp",
    ],
)

history_df = (
    history_df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


print(
    f"\nForecast dataset: {forecast_df.shape}"
)

print(
    f"History dataset : {history_df.shape}"
)


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
# 6. CREATE PHYSICS FEATURES
# ----------------------------------------------------------------

print(
    "\nCreating physics-informed features..."
)


# Solar elevation
history_df[
    "solar_elevation_deg"
] = (
        90.0
        - history_df["solar_zenith_angle"]
)


# Sun cosine of zenith
zenith_rad = np.deg2rad(
    history_df["solar_zenith_angle"]
)

history_df[
    "sun_cos_zenith"
] = np.maximum(
    np.cos(zenith_rad),
    0.0
)


# Daylight flag
history_df[
    "daylight_flag"
] = (
        history_df["solar_zenith_angle"] < 90
).astype(int)


# Relative air mass
zenith_for_airmass = (
    history_df["solar_zenith_angle"]
    .clip(
        lower=0,
        upper=89.9,
    )
)

relative_airmass = (
    pvlib.atmosphere
    .get_relative_airmass(
        zenith_for_airmass,
        model="kastenyoung1989",
    )
)

relative_airmass = pd.Series(
    relative_airmass,
    index=history_df.index,
)

relative_airmass.loc[
    history_df["daylight_flag"] == 0
    ] = 0

history_df[
    "relative_airmass"
] = relative_airmass


# Pressure-corrected air mass
history_df[
    "pressure_corrected_airmass"
] = (
        history_df["relative_airmass"]
        * history_df["PS"]
        / 1013.25
)


# Module - ambient temperature
history_df[
    "module_ambient_delta"
] = (
        history_df["module_temp__5063"]
        - history_df["ambient_temp__5062"]
)


# POA × module temperature
history_df[
    "poa_module_temp_interaction"
] = (
        history_df["poa_irradiance__5061"]
        * history_df["module_temp__5063"]
)


# POA × ambient temperature
history_df[
    "poa_ambient_temp_interaction"
] = (
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


v19_features = (
        base_features
        + physics_features
)


print(
    f"\nBase features     : "
    f"{len(base_features)}"
)

print(
    f"Physics features  : "
    f"{len(physics_features)}"
)

print(
    f"Total V19 features: "
    f"{len(v19_features)}"
)


# ----------------------------------------------------------------
# 7. MERGE PHYSICS FEATURES
# ----------------------------------------------------------------

physics_source = history_df[
    ["timestamp"] + physics_features
    ].copy()


forecast_df = forecast_df.merge(
    physics_source,
    on="timestamp",
    how="left",
    validate="one_to_one",
)


# ----------------------------------------------------------------
# 8. CHECK MISSING VALUES
# ----------------------------------------------------------------

missing_features = (
    forecast_df[
        v19_features
    ]
    .isna()
    .sum()
)


print(
    "\nMissing V19 feature values:"
)

print(
    missing_features[
        missing_features > 0
        ]
)


if missing_features.sum() > 0:

    raise ValueError(
        "Missing feature values detected."
    )


# ----------------------------------------------------------------
# 9. CREATE ACTIVE TARGET
# ----------------------------------------------------------------

forecast_df[
    "active_target"
] = (
        forecast_df[target_col]
        > ACTIVE_THRESHOLD
).astype(int)


# ----------------------------------------------------------------
# 10. ROLLING FOLDS
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


# ----------------------------------------------------------------
# 11. METRICS
# ----------------------------------------------------------------

def regression_metrics(
        y_true,
        y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return mae, rmse, r2


# ----------------------------------------------------------------
# 12. RESULTS
# ----------------------------------------------------------------

results = []


# ----------------------------------------------------------------
# 13. RUN FOLDS
# ----------------------------------------------------------------

for fold_number, fold in enumerate(
        folds,
        start=1,
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"Fold {fold_number}: "
        f"TRAIN "
        f"{fold['train_start'][:4]}-"
        f"{fold['train_end'][:4]} "
        f"→ VALIDATE {fold['year']}"
    )

    print(
        "-" * 70
    )


    # ------------------------------------------------------------
    # TRAIN / VALIDATION
    # ------------------------------------------------------------

    train_mask = (

            (
                    forecast_df[
                        "target_timestamp"
                    ]
                    >= pd.Timestamp(
                fold["train_start"]
            )
            )

            &

            (
                    forecast_df[
                        "target_timestamp"
                    ]
                    <= pd.Timestamp(
                fold["train_end"]
            )
            )

            &

            forecast_df[
                target_col
            ].notna()
    )


    val_mask = (

            (
                    forecast_df[
                        "target_timestamp"
                    ]
                    >= pd.Timestamp(
                fold["val_start"]
            )
            )

            &

            (
                    forecast_df[
                        "target_timestamp"
                    ]
                    <= pd.Timestamp(
                fold["val_end"]
            )
            )

            &

            forecast_df[
                target_col
            ].notna()
    )


    train_data = forecast_df.loc[
        train_mask
    ].copy()

    val_data = forecast_df.loc[
        val_mask
    ].copy()


    print(
        f"Train rows      : "
        f"{len(train_data)}"
    )

    print(
        f"Validation rows : "
        f"{len(val_data)}"
    )


    # ============================================================
    # STAGE 1 - CLASSIFIER
    # ============================================================

    print(
        "\nSTAGE 1 - ACTIVE / NEAR-ZERO CLASSIFIER"
    )


    X_train_cls = train_data[
        v19_features
    ]

    y_train_cls = train_data[
        "active_target"
    ]

    X_val_cls = val_data[
        v19_features
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
            (
                X_val_cls,
                y_val_cls,
            )
        ],

        eval_metric="binary_logloss",

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False,
            )
        ],
    )


    # Classification probability
    val_probability = (
        classifier
        .predict_proba(
            X_val_cls
        )[:, 1]
    )


    val_active_prediction = (
            val_probability
            >= CLASSIFICATION_THRESHOLD
    ).astype(int)


    cls_accuracy = (
        accuracy_score(
            y_val_cls,
            val_active_prediction,
        )
    )


    cls_precision = (
        precision_score(
            y_val_cls,
            val_active_prediction,
            zero_division=0,
        )
    )


    cls_recall = (
        recall_score(
            y_val_cls,
            val_active_prediction,
            zero_division=0,
        )
    )


    cls_f1 = (
        f1_score(
            y_val_cls,
            val_active_prediction,
            zero_division=0,
        )
    )


    print(
        f"Best classifier iteration: "
        f"{classifier.best_iteration_}"
    )

    print(
        f"Accuracy : "
        f"{cls_accuracy:.4f}"
    )

    print(
        f"F1       : "
        f"{cls_f1:.4f}"
    )


    # ============================================================
    # STAGE 2 - LOG ACTIVE POWER REGRESSOR
    # ============================================================

    print(
        "\nSTAGE 2 - LOG ACTIVE POWER REGRESSOR"
    )


    active_train = train_data[
        train_data[
            "active_target"
        ] == 1
        ].copy()


    active_val = val_data[
        val_data[
            "active_target"
        ] == 1
        ].copy()


    X_train_reg = active_train[
        v19_features
    ]

    y_train_reg = active_train[
        target_col
    ]


    X_val_reg = active_val[
        v19_features
    ]

    y_val_reg = active_val[
        target_col
    ]


    # ------------------------------------------------------------
    # Transform target
    # ------------------------------------------------------------

    y_train_log = np.log1p(
        y_train_reg
    )

    y_val_log = np.log1p(
        y_val_reg
    )


    print(
        f"\nActive training rows   : "
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

        y_train_log,

        eval_set=[
            (
                X_val_reg,
                y_val_log,
            )
        ],

        eval_metric="rmse",

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False,
            )
        ],
    )


    print(
        f"\nBest regressor iteration: "
        f"{regressor.best_iteration_}"
    )


    # ------------------------------------------------------------
    # LOG prediction
    # ------------------------------------------------------------

    log_prediction = (
        regressor.predict(
            X_val_cls
        )
    )


    # Inverse transformation
    regression_prediction = (
        np.expm1(
            log_prediction
        )
    )


    # Remove tiny negative numerical values
    regression_prediction = np.maximum(
        regression_prediction,
        0.0,
    )


    # ============================================================
    # FINAL TWO-STAGE PREDICTION
    # ============================================================

    final_prediction = np.where(

        val_active_prediction == 1,

        regression_prediction,

        0.0,
        )


    final_prediction = np.maximum(
        final_prediction,
        0.0,
    )


    # ============================================================
    # FINAL METRICS
    # ============================================================

    final_mae, final_rmse, final_r2 = (
        regression_metrics(
            val_data[
                target_col
            ],
            final_prediction,
        )
    )


    # ------------------------------------------------------------
    # Active-only regressor metrics
    # ------------------------------------------------------------

    active_mae, active_rmse, active_r2 = (
        regression_metrics(
            y_val_reg,
            regression_prediction[
                val_data[
                    "active_target"
                ].to_numpy() == 1
                ],
        )
    )


    print(
        "\nV19 Results:"
    )

    print(
        f"MAE : "
        f"{final_mae:.4f}"
    )

    print(
        f"RMSE: "
        f"{final_rmse:.4f}"
    )

    print(
        f"R²  : "
        f"{final_r2:.4f}"
    )


    print(
        "\nActive-only log-regressor:"
    )

    print(
        f"MAE : "
        f"{active_mae:.4f}"
    )

    print(
        f"RMSE: "
        f"{active_rmse:.4f}"
    )

    print(
        f"R²  : "
        f"{active_r2:.4f}"
    )


    results.append({

        "fold":
            fold_number,

        "validation_year":
            fold["year"],

        "train_rows":
            len(train_data),

        "validation_rows":
            len(val_data),

        "active_train_rows":
            len(active_train),

        "active_validation_rows":
            len(active_val),

        "classifier_accuracy":
            cls_accuracy,

        "classifier_precision":
            cls_precision,

        "classifier_recall":
            cls_recall,

        "classifier_f1":
            cls_f1,

        "classifier_best_iteration":
            classifier.best_iteration_,

        "regressor_best_iteration":
            regressor.best_iteration_,

        "active_mae":
            active_mae,

        "active_rmse":
            active_rmse,

        "active_r2":
            active_r2,

        "final_mae":
            final_mae,

        "final_rmse":
            final_rmse,

        "final_r2":
            final_r2,
    })


# ----------------------------------------------------------------
# 14. SUMMARY
# ----------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


print(
    "\n" + "=" * 70
)

print(
    "MODEL 1 V19 ROLLING VALIDATION SUMMARY"
)

print(
    "=" * 70
)


print(
    results_df.to_string(
        index=False
    )
)


avg_mae = (
    results_df[
        "final_mae"
    ].mean()
)

avg_rmse = (
    results_df[
        "final_rmse"
    ].mean()
)

avg_r2 = (
    results_df[
        "final_r2"
    ].mean()
)

avg_f1 = (
    results_df[
        "classifier_f1"
    ].mean()
)


print(
    "\nAverage V19 performance:"
)

print(
    f"MAE : "
    f"{avg_mae:.4f}"
)

print(
    f"RMSE: "
    f"{avg_rmse:.4f}"
)

print(
    f"R²  : "
    f"{avg_r2:.4f}"
)

print(
    f"Stage-1 F1: "
    f"{avg_f1:.4f}"
)


# ----------------------------------------------------------------
# 15. REFERENCES
# ----------------------------------------------------------------

print(
    "\nReference - V14:"
)

print(
    "MAE : 14.3198"
)

print(
    "RMSE: 29.6485"
)

print(
    "R²  : 0.9124"
)


print(
    "\nReference - V15:"
)

print(
    "MAE : 14.1381"
)

print(
    "RMSE: 29.6294"
)

print(
    "R²  : 0.9125"
)


print(
    "\nReference - V18 (365d):"
)

print(
    "MAE : 14.0155"
)

print(
    "RMSE: 29.4342"
)

print(
    "R²  : 0.9138"
)


# ----------------------------------------------------------------
# 16. SAVE RESULTS
# ----------------------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\nValidation results saved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\n" + "=" * 70
)

print(
    "V19 COMPLETE"
)

print(
    "=" * 70
)