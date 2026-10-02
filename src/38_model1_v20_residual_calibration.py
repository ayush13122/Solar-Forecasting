# ================================================================
# STEP 74 - MODEL 1 V20: RESIDUAL CALIBRATION
# ================================================================
#
# Base model:
#   V15 Two-Stage LightGBM
#
# Stage 1:
#   Active / Near-zero classifier
#
# Stage 2:
#   Active-power regressor
#
# V20 addition:
#   Small residual-correction model
#
# Important:
#   Residual model is trained on OUT-OF-SAMPLE predictions created
#   using an inner chronological split inside each outer training
#   fold.
#
# Outer rolling validation:
#   2012-2013 -> 2014
#   2013-2014 -> 2015
#   2014-2015 -> 2016
#
# 2017-2018 is NOT used.
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
)


print("=" * 70)
print("STEP 74 - MODEL 1 V20: RESIDUAL CALIBRATION")
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
        / "Model1_V20_Residual_Calibration_Validation.csv"
)


# ----------------------------------------------------------------
# 3. SETTINGS
# ----------------------------------------------------------------

ACTIVE_THRESHOLD = 1.0

CLASSIFICATION_THRESHOLD = 0.50

# Inner split:
# first 70% = base-model fitting
# final 30% = honest residual-generation block

INNER_TRAIN_RATIO = 0.70


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
# 6. CREATE V14/V15 PHYSICS FEATURES
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


# Cosine of solar zenith
zenith_rad = np.deg2rad(
    history_df["solar_zenith_angle"]
)

history_df[
    "sun_cos_zenith"
] = np.maximum(
    np.cos(zenith_rad),
    0.0,
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
        history_df[
            "relative_airmass"
        ]
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


v20_features = (
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
    f"Total V20 features: "
    f"{len(v20_features)}"
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
        v20_features
    ]
    .isna()
    .sum()
)


print(
    "\nMissing V20 feature values:"
)

print(
    missing_features[
        missing_features > 0
        ]
)


if missing_features.sum() > 0:

    raise ValueError(
        "Missing V20 feature values detected."
    )


# ----------------------------------------------------------------
# 9. CREATE ACTIVE TARGET
# ----------------------------------------------------------------

forecast_df[
    "active_target"
] = (
        forecast_df[
            target_col
        ]
        > ACTIVE_THRESHOLD
).astype(int)


# ----------------------------------------------------------------
# 10. OUTER ROLLING FOLDS
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
# 11. CORRECTION FEATURES
# ----------------------------------------------------------------
# Deliberately smaller than the full 34-feature base model.
#
# These features are used to explain systematic residual bias,
# not to build a second full forecasting engine.

correction_features = [

    "base_prediction",

    "stage1_probability",

    "poa_irradiance__5061",

    "module_temp__5063",

    "ambient_temp__5062",

    "solar_zenith_angle",

    "CLOUD_AMT",

    "module_ambient_delta",

    "solar_elevation_deg",

    "hour_sin",

    "hour_cos",
]


print(
    "\nResidual-correction features:"
)

print(
    correction_features
)


# ----------------------------------------------------------------
# 12. METRICS
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
# 13. BASE TWO-STAGE MODEL FUNCTION
# ----------------------------------------------------------------

def train_two_stage_model(
        train_data,
        predict_data,
        classifier_estimators=None,
        regressor_estimators=None,
        use_early_stopping=False,
        eval_data=None,
):
    """
    Train V15-style two-stage model.

    If use_early_stopping=True:
        eval_data must be supplied.

    Otherwise fixed estimators are used.
    """

    # ------------------------------------------------------------
    # Stage 1 - classifier
    # ------------------------------------------------------------

    classifier = lgb.LGBMClassifier(

        objective="binary",

        n_estimators=(
            1000
            if classifier_estimators is None
            else classifier_estimators
        ),

        learning_rate=0.05,

        num_leaves=31,

        max_depth=-1,

        subsample=0.9,

        colsample_bytree=0.9,

        random_state=42,

        n_jobs=-1,
    )


    if (
            use_early_stopping
            and
            eval_data is not None
    ):

        classifier.fit(

            train_data[
                v20_features
            ],

            train_data[
                "active_target"
            ],

            eval_set=[
                (
                    eval_data[
                        v20_features
                    ],

                    eval_data[
                        "active_target"
                    ],
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

    else:

        classifier.fit(

            train_data[
                v20_features
            ],

            train_data[
                "active_target"
            ],
        )


    # ------------------------------------------------------------
    # Stage 2 - active-power regressor
    # ------------------------------------------------------------

    active_train = train_data[
        train_data[
            "active_target"
        ] == 1
        ].copy()


    regressor = lgb.LGBMRegressor(

        objective="regression",

        n_estimators=(
            1000
            if regressor_estimators is None
            else regressor_estimators
        ),

        learning_rate=0.05,

        num_leaves=31,

        max_depth=-1,

        subsample=0.9,

        colsample_bytree=0.9,

        random_state=42,

        n_jobs=-1,
    )


    if (
            use_early_stopping
            and
            eval_data is not None
    ):

        active_eval = eval_data[
            eval_data[
                "active_target"
            ] == 1
            ].copy()


        regressor.fit(

            active_train[
                v20_features
            ],

            active_train[
                target_col
            ],

            eval_set=[
                (
                    active_eval[
                        v20_features
                    ],

                    active_eval[
                        target_col
                    ],
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

    else:

        regressor.fit(

            active_train[
                v20_features
            ],

            active_train[
                target_col
            ],
        )


    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    probability = (
        classifier
        .predict_proba(
            predict_data[
                v20_features
            ]
        )[:, 1]
    )


    active_prediction = (
            probability
            >= CLASSIFICATION_THRESHOLD
    ).astype(int)


    regression_prediction = (
        regressor.predict(
            predict_data[
                v20_features
            ]
        )
    )


    regression_prediction = np.maximum(
        regression_prediction,
        0.0,
    )


    final_prediction = np.where(

        active_prediction == 1,

        regression_prediction,

        0.0,
        )


    final_prediction = np.maximum(
        final_prediction,
        0.0,
    )


    best_classifier_iteration = (
        getattr(
            classifier,
            "best_iteration_",
            None,
        )
    )


    best_regressor_iteration = (
        getattr(
            regressor,
            "best_iteration_",
            None,
        )
    )


    return (
        final_prediction,
        probability,
        best_classifier_iteration,
        best_regressor_iteration,
    )


# ----------------------------------------------------------------
# 14. RESULTS
# ----------------------------------------------------------------

results = []


# ----------------------------------------------------------------
# 15. OUTER FOLDS
# ----------------------------------------------------------------

for fold_number, fold in enumerate(
        folds,
        start=1,
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"OUTER FOLD {fold_number}: "
        f"{fold['year']}"
    )

    print(
        "=" * 70
    )


    # ------------------------------------------------------------
    # Outer train / validation
    # ------------------------------------------------------------

    outer_train_mask = (

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


    outer_val_mask = (

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


    outer_train = forecast_df.loc[
        outer_train_mask
    ].copy()


    outer_val = forecast_df.loc[
        outer_val_mask
    ].copy()


    print(
        f"Outer train rows: "
        f"{len(outer_train)}"
    )

    print(
        f"Outer validation rows: "
        f"{len(outer_val)}"
    )


    # ============================================================
    # INNER CHRONOLOGICAL SPLIT
    # ============================================================

    split_position = int(
        len(outer_train)
        * INNER_TRAIN_RATIO
    )


    inner_train = (
        outer_train
        .sort_values(
            "target_timestamp"
        )
        .iloc[
            :split_position
        ]
        .copy()
    )


    inner_calibration = (
        outer_train
        .sort_values(
            "target_timestamp"
        )
        .iloc[
            split_position:
        ]
        .copy()
    )


    print(
        "\nInner split:"
    )

    print(
        f"Base-fit rows       : "
        f"{len(inner_train)}"
    )

    print(
        f"Residual rows       : "
        f"{len(inner_calibration)}"
    )


    # ============================================================
    # STEP A - TRAIN BASE MODEL ON EARLIER INNER DATA
    # ============================================================

    print(
        "\nTraining inner base model..."
    )


    (
        inner_base_prediction,
        inner_probability,
        inner_cls_iteration,
        inner_reg_iteration,
    ) = train_two_stage_model(

        train_data=inner_train,

        predict_data=inner_calibration,

        use_early_stopping=True,

        eval_data=inner_calibration,
    )


    print(
        f"Inner classifier best iteration: "
        f"{inner_cls_iteration}"
    )

    print(
        f"Inner regressor best iteration: "
        f"{inner_reg_iteration}"
    )


    # ============================================================
    # STEP B - CREATE HONEST RESIDUALS
    # ============================================================

    inner_actual = (
        inner_calibration[
            target_col
        ]
        .to_numpy()
    )


    inner_residual = (
            inner_actual
            - inner_base_prediction
    )


    calibration_data = inner_calibration.copy()


    calibration_data[
        "base_prediction"
    ] = inner_base_prediction


    calibration_data[
        "stage1_probability"
    ] = inner_probability


    calibration_data[
        "residual"
    ] = inner_residual


    print(
        "\nResidual statistics:"
    )

    print(
        f"Mean residual : "
        f"{inner_residual.mean():.4f}"
    )

    print(
        f"Median residual : "
        f"{np.median(inner_residual):.4f}"
    )

    print(
        f"Std residual : "
        f"{inner_residual.std():.4f}"
    )


    # ============================================================
    # STEP C - TRAIN RESIDUAL CORRECTION MODEL
    # ============================================================

    print(
        "\nTraining residual correction model..."
    )


    correction_model = lgb.LGBMRegressor(

        objective="regression",

        n_estimators=150,

        learning_rate=0.03,

        num_leaves=15,

        max_depth=4,

        min_child_samples=80,

        subsample=0.9,

        colsample_bytree=0.9,

        random_state=42,

        n_jobs=-1,
    )


    correction_model.fit(

        calibration_data[
            correction_features
        ],

        calibration_data[
            "residual"
        ],
    )


    # ============================================================
    # STEP D - REFIT BASE MODEL ON ALL OUTER TRAIN DATA
    # ============================================================

    print(
        "\nRefitting base model on complete outer training data..."
    )


    # Use inner-selected iterations
    final_classifier_estimators = (
        inner_cls_iteration
        if inner_cls_iteration is not None
        else 150
    )


    final_regressor_estimators = (
        inner_reg_iteration
        if inner_reg_iteration is not None
        else 150
    )


    (
        outer_base_prediction,
        outer_probability,
        _,
        _,
    ) = train_two_stage_model(

        train_data=outer_train,

        predict_data=outer_val,

        classifier_estimators=
        final_classifier_estimators,

        regressor_estimators=
        final_regressor_estimators,

        use_early_stopping=False,
    )


    # ============================================================
    # STEP E - APPLY RESIDUAL CORRECTION
    # ============================================================

    correction_input = outer_val.copy()


    correction_input[
        "base_prediction"
    ] = outer_base_prediction


    correction_input[
        "stage1_probability"
    ] = outer_probability


    correction_prediction = (
        correction_model.predict(
            correction_input[
                correction_features
            ]
        )
    )


    final_prediction = (
            outer_base_prediction
            + correction_prediction
    )


    # Physical lower bound
    final_prediction = np.maximum(
        final_prediction,
        0.0,
    )


    # Upper bound based on observed plant AC range
    final_prediction = np.minimum(
        final_prediction,
        400.0,
    )


    # ============================================================
    # STEP F - EVALUATE
    # ============================================================

    y_val = (
        outer_val[
            target_col
        ].to_numpy()
    )


    base_mae, base_rmse, base_r2 = (
        regression_metrics(
            y_val,
            outer_base_prediction,
        )
    )


    corrected_mae, corrected_rmse, corrected_r2 = (
        regression_metrics(
            y_val,
            final_prediction,
        )
    )


    correction_mean = (
        correction_prediction.mean()
    )

    correction_abs_mean = (
        np.abs(
            correction_prediction
        ).mean()
    )


    print(
        "\nBASE V15-STYLE RESULT:"
    )

    print(
        f"MAE : "
        f"{base_mae:.4f}"
    )

    print(
        f"RMSE: "
        f"{base_rmse:.4f}"
    )

    print(
        f"R²  : "
        f"{base_r2:.4f}"
    )


    print(
        "\nV20 CORRECTED RESULT:"
    )

    print(
        f"MAE : "
        f"{corrected_mae:.4f}"
    )

    print(
        f"RMSE: "
        f"{corrected_rmse:.4f}"
    )

    print(
        f"R²  : "
        f"{corrected_r2:.4f}"
    )


    print(
        "\nCorrection statistics:"
    )

    print(
        f"Mean correction      : "
        f"{correction_mean:.4f}"
    )

    print(
        f"Mean absolute correction: "
        f"{correction_abs_mean:.4f}"
    )


    results.append({

        "fold":
            fold_number,

        "validation_year":
            fold["year"],

        "outer_train_rows":
            len(outer_train),

        "outer_validation_rows":
            len(outer_val),

        "inner_train_rows":
            len(inner_train),

        "inner_calibration_rows":
            len(inner_calibration),

        "inner_classifier_best_iteration":
            inner_cls_iteration,

        "inner_regressor_best_iteration":
            inner_reg_iteration,

        "base_mae":
            base_mae,

        "base_rmse":
            base_rmse,

        "base_r2":
            base_r2,

        "corrected_mae":
            corrected_mae,

        "corrected_rmse":
            corrected_rmse,

        "corrected_r2":
            corrected_r2,

        "mean_correction":
            correction_mean,

        "mean_abs_correction":
            correction_abs_mean,
    })


# ----------------------------------------------------------------
# 16. SUMMARY
# ----------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


print(
    "\n" + "=" * 70
)

print(
    "MODEL 1 V20 ROLLING VALIDATION SUMMARY"
)

print(
    "=" * 70
)


print(
    results_df.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 17. AVERAGES
# ----------------------------------------------------------------

avg_base_mae = (
    results_df[
        "base_mae"
    ].mean()
)

avg_base_rmse = (
    results_df[
        "base_rmse"
    ].mean()
)

avg_base_r2 = (
    results_df[
        "base_r2"
    ].mean()
)


avg_corrected_mae = (
    results_df[
        "corrected_mae"
    ].mean()
)

avg_corrected_rmse = (
    results_df[
        "corrected_rmse"
    ].mean()
)

avg_corrected_r2 = (
    results_df[
        "corrected_r2"
    ].mean()
)


print(
    "\nAverage BASE performance:"
)

print(
    f"MAE : "
    f"{avg_base_mae:.4f}"
)

print(
    f"RMSE: "
    f"{avg_base_rmse:.4f}"
)

print(
    f"R²  : "
    f"{avg_base_r2:.4f}"
)


print(
    "\nAverage V20 CORRECTED performance:"
)

print(
    f"MAE : "
    f"{avg_corrected_mae:.4f}"
)

print(
    f"RMSE: "
    f"{avg_corrected_rmse:.4f}"
)

print(
    f"R²  : "
    f"{avg_corrected_r2:.4f}"
)


# ----------------------------------------------------------------
# 18. REFERENCES
# ----------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "REFERENCE RESULTS"
)

print(
    "=" * 70
)

print(
    "\nV15:"
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
    "\nV18 (365 days):"
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
# 19. SAVE
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
    "V20 COMPLETE"
)

print(
    "=" * 70
)