# Step 68 — V16 LSTM
# ================================================================
# STEP 68 - MODEL 1 V16: LSTM SEQUENCE FORECASTING
# ================================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as npS
import pandas as pd
import tensorflow as tf
import pvlib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ----------------------------------------------------------------
# 1. REPRODUCIBILITY
# ----------------------------------------------------------------

SEED = 42

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


print("=" * 70)
print("STEP 68 - MODEL 1 V16: LSTM SEQUENCE FORECASTING")
print("=" * 70)


# ----------------------------------------------------------------
# 2. FIND ACTUAL PROJECT ROOT
# ----------------------------------------------------------------
# We intentionally require .venv + known processed dataset.
# This avoids accidentally treating nested src folders as root.

current_path = Path(__file__).resolve()

PROJECT_ROOT = None

KNOWN_DATA_FILE = (
    "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)

for parent in current_path.parents:

    has_venv = (parent / ".venv").exists()

    has_data = (
            parent
            / "data"
            / "processed"
            / KNOWN_DATA_FILE
    ).exists()

    if has_venv and has_data:
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Actual project root not found.\n"
        "Expected a parent folder containing:\n"
        "1. .venv\n"
        "2. data/processed/"
        f"{KNOWN_DATA_FILE}"
    )

print(f"\nActual project root: {PROJECT_ROOT}")


# ----------------------------------------------------------------
# 3. INPUT / OUTPUT PATHS
# ----------------------------------------------------------------

HISTORY_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)

OUTPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V16_LSTM_Validation.csv"
)


# ----------------------------------------------------------------
# 4. SETTINGS
# ----------------------------------------------------------------

LOOKBACK = 24

BATCH_SIZE = 256

EPOCHS = 50

PATIENCE = 8


# ----------------------------------------------------------------
# 5. LOAD FULL HOURLY HISTORY
# ----------------------------------------------------------------

df = pd.read_csv(
    HISTORY_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


print(f"\nHistory dataset: {df.shape}")

print(
    f"Date range: "
    f"{df['timestamp'].min()} → "
    f"{df['timestamp'].max()}"
)


# ----------------------------------------------------------------
# 6. BASE FEATURES
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

target_col = "ac_power__5069"


# ----------------------------------------------------------------
# 7. CREATE PHYSICS FEATURES
# ----------------------------------------------------------------

print("\nCreating physics-informed features...")


# Solar elevation
df["solar_elevation_deg"] = (
        90.0 - df["solar_zenith_angle"]
)


# Cosine of solar zenith
zenith_rad = np.deg2rad(
    df["solar_zenith_angle"]
)

df["sun_cos_zenith"] = np.maximum(
    np.cos(zenith_rad),
    0.0
)


# Daylight flag
df["daylight_flag"] = (
        df["solar_zenith_angle"] < 90
).astype(int)


# Relative air mass
zenith_for_airmass = (
    df["solar_zenith_angle"]
    .clip(lower=0, upper=89.9)
)

relative_airmass = pvlib.atmosphere.get_relative_airmass(
    zenith_for_airmass,
    model="kastenyoung1989"
)

relative_airmass = pd.Series(
    relative_airmass,
    index=df.index
)

relative_airmass.loc[
    df["daylight_flag"] == 0
    ] = 0

df["relative_airmass"] = relative_airmass


# Pressure-corrected air mass
df["pressure_corrected_airmass"] = (
        df["relative_airmass"]
        * df["PS"]
        / 1013.25
)


# Module - ambient temperature difference
df["module_ambient_delta"] = (
        df["module_temp__5063"]
        - df["ambient_temp__5062"]
)


# POA × module temperature
df["poa_module_temp_interaction"] = (
        df["poa_irradiance__5061"]
        * df["module_temp__5063"]
)


# POA × ambient temperature
df["poa_ambient_temp_interaction"] = (
        df["poa_irradiance__5061"]
        * df["ambient_temp__5062"]
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


features = base_features + physics_features


print(f"\nBase features    : {len(base_features)}")
print(f"Physics features : {len(physics_features)}")
print(f"Total features   : {len(features)}")


# ----------------------------------------------------------------
# 8. CHECK FEATURE / TARGET MISSING VALUES
# ----------------------------------------------------------------

feature_missing = df[features].isna().sum()

target_missing = df[target_col].isna().sum()

print(
    f"\nTotal missing feature values: "
    f"{feature_missing.sum()}"
)

print(
    f"Missing target values: "
    f"{target_missing}"
)


if feature_missing.sum() > 0:

    print("\nMissing feature columns:")
    print(
        feature_missing[
            feature_missing > 0
            ]
    )

    raise ValueError(
        "Unexpected missing values found in model features."
    )


# ----------------------------------------------------------------
# 9. CREATE NEXT-HOUR TARGET
# ----------------------------------------------------------------

df["target_ac_power"] = (
    df[target_col].shift(-1)
)

df["target_timestamp"] = (
    df["timestamp"].shift(-1)
)


# Require exactly one-hour transition
df["valid_next_hour_pair"] = (
        (
                df["target_timestamp"]
                - df["timestamp"]
        )
        == pd.Timedelta(hours=1)
)


print(
    "\nValid next-hour pairs:",
    int(df["valid_next_hour_pair"].sum())
)

print(
    "Invalid next-hour pairs:",
    int(
        (~df["valid_next_hour_pair"]).sum()
    )
)


# ----------------------------------------------------------------
# 10. PREPARE ARRAYS
# ----------------------------------------------------------------

feature_values = (
    df[features]
    .to_numpy(dtype=np.float32)
)

target_values = (
    df["target_ac_power"]
    .to_numpy(dtype=np.float32)
)

timestamps = (
    df["timestamp"]
    .to_numpy()
)

target_timestamps = (
    df["target_timestamp"]
    .to_numpy()
)

valid_pair = (
    df["valid_next_hour_pair"]
    .to_numpy()
)

target_available = ~np.isnan(
    target_values
)


# ----------------------------------------------------------------
# 11. BUILD 24-HOUR SEQUENCES
# ----------------------------------------------------------------

print(
    f"\nBuilding {LOOKBACK}-hour sequences..."
)


X_sequences = []
y_sequences = []

sequence_end_timestamps = []
sequence_target_timestamps = []


for i in range(
        LOOKBACK - 1,
        len(df) - 1
):

    start_idx = i - LOOKBACK + 1

    # ------------------------------------------------------------
    # Require complete 24-hour hourly history
    # ------------------------------------------------------------

    elapsed_history = (
            timestamps[i]
            - timestamps[start_idx]
    )

    if elapsed_history != np.timedelta64(
            LOOKBACK - 1,
            "h"
    ):
        continue


    # ------------------------------------------------------------
    # Require valid next-hour target
    # ------------------------------------------------------------

    if not valid_pair[i]:
        continue

    if not target_available[i]:
        continue


    sequence = feature_values[
        start_idx : i + 1
    ]


    X_sequences.append(sequence)

    y_sequences.append(
        target_values[i]
    )

    sequence_end_timestamps.append(
        timestamps[i]
    )

    sequence_target_timestamps.append(
        target_timestamps[i]
    )


X = np.asarray(
    X_sequences,
    dtype=np.float32
)

y = np.asarray(
    y_sequences,
    dtype=np.float32
)

sequence_end_timestamps = pd.to_datetime(
    sequence_end_timestamps
)

sequence_target_timestamps = pd.to_datetime(
    sequence_target_timestamps
)


print(
    f"\nSequence dataset shape: {X.shape}"
)

print(
    f"Target shape          : {y.shape}"
)


# ----------------------------------------------------------------
# 12. SANITY CHECK
# ----------------------------------------------------------------

if len(X) == 0:
    raise ValueError(
        "No valid sequences were generated."
    )


print(
    "\nExpected sequence dimensions:"
)

print(
    f"Samples × Timesteps × Features = "
    f"{X.shape[0]} × {X.shape[1]} × {X.shape[2]}"
)


# ----------------------------------------------------------------
# 13. DEFINE ROLLING FOLDS
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
# 14. METRICS
# ----------------------------------------------------------------

def calculate_metrics(
        y_true,
        y_pred
):

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


results = []


# ----------------------------------------------------------------
# 15. TRAIN EACH FOLD
# ----------------------------------------------------------------

for fold_number, fold in enumerate(
        folds,
        start=1
):

    print("\n" + "-" * 70)

    print(
        f"Fold {fold_number}: "
        f"TRAIN {fold['train_start'][:4]}-"
        f"{fold['train_end'][:4]} "
        f"→ VALIDATE {fold['year']}"
    )

    print("-" * 70)


    # ------------------------------------------------------------
    # Split using TARGET timestamp
    # ------------------------------------------------------------

    train_mask = (
            (sequence_target_timestamps
             >= pd.Timestamp(fold["train_start"]))
            &
            (sequence_target_timestamps
             <= pd.Timestamp(fold["train_end"]))
    )

    val_mask = (
            (sequence_target_timestamps
             >= pd.Timestamp(fold["val_start"]))
            &
            (sequence_target_timestamps
             <= pd.Timestamp(fold["val_end"]))
    )


    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()

    X_val = X[val_mask].copy()
    y_val = y[val_mask].copy()


    print(
        f"Train sequences      : "
        f"{len(X_train)}"
    )

    print(
        f"Validation sequences : "
        f"{len(X_val)}"
    )


    # ------------------------------------------------------------
    # Scale features using TRAIN ONLY
    # ------------------------------------------------------------

    n_features = X_train.shape[2]

    feature_scaler = StandardScaler()

    X_train_2d = X_train.reshape(
        -1,
        n_features
    )

    feature_scaler.fit(
        X_train_2d
    )

    X_train = feature_scaler.transform(
        X_train_2d
    ).reshape(
        X_train.shape
    ).astype(np.float32)


    X_val_2d = X_val.reshape(
        -1,
        n_features
    )

    X_val = feature_scaler.transform(
        X_val_2d
    ).reshape(
        X_val.shape
    ).astype(np.float32)


    # ------------------------------------------------------------
    # Scale target using TRAIN ONLY
    # ------------------------------------------------------------

    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(
        y_train.reshape(-1, 1)
    ).ravel().astype(np.float32)


    y_val_scaled = target_scaler.transform(
        y_val.reshape(-1, 1)
    ).ravel().astype(np.float32)


    # ------------------------------------------------------------
    # Clear previous TensorFlow model
    # ------------------------------------------------------------

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(SEED)


    # ------------------------------------------------------------
    # LSTM MODEL
    # ------------------------------------------------------------

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(
                LOOKBACK,
                n_features
            )
        ),

        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.LSTM(
            32
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1
        ),
    ])


    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="mse",

        metrics=[
            tf.keras.metrics.MeanAbsoluteError()
        ],
    )


    print("\nLSTM architecture:")

    model.summary()


    # ------------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=PATIENCE,

        restore_best_weights=True,

        verbose=1,
    )


    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=4,

        min_lr=1e-5,

        verbose=1,
    )


    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    history = model.fit(

        X_train,

        y_train_scaled,

        validation_data=(
            X_val,
            y_val_scaled
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=False,

        callbacks=[
            early_stopping,
            reduce_lr,
        ],

        verbose=1,
    )


    # ------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------

    pred_scaled = model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()


    # Inverse transform to kW
    y_pred = target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()


    # Prevent physically impossible negative prediction
    y_pred = np.maximum(
        y_pred,
        0.0
    )


    # ------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------

    mae, rmse, r2 = calculate_metrics(
        y_val,
        y_pred
    )


    best_epoch = (
            np.argmin(
                history.history["val_loss"]
            ) + 1
    )


    print("\nV16 Results:")

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )


    results.append({

        "fold": fold_number,

        "validation_year": fold["year"],

        "train_sequences": len(X_train),

        "validation_sequences": len(X_val),

        "best_epoch": best_epoch,

        "mae": mae,

        "rmse": rmse,

        "r2": r2,
    })


# ----------------------------------------------------------------
# 16. SUMMARY
# ----------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print(
    "MODEL 1 V16 LSTM ROLLING VALIDATION SUMMARY"
)
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


avg_mae = results_df["mae"].mean()

avg_rmse = results_df["rmse"].mean()

avg_r2 = results_df["r2"].mean()


print("\nAverage V16 performance:")

print(
    f"MAE : {avg_mae:.4f}"
)

print(
    f"RMSE: {avg_rmse:.4f}"
)

print(
    f"R²  : {avg_r2:.4f}"
)


# ----------------------------------------------------------------
# 17. REFERENCES
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

print("\nReference - V15:")
print("MAE : 14.1381")
print("RMSE: 29.6294")
print("R²  : 0.9125")


# ----------------------------------------------------------------
# 18. SAVE RESULTS
# ----------------------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nValidation results saved to:"
)

print(
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("V16 COMPLETE")
print("=" * 70)