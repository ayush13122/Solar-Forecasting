# Step 66 — V14 Physics-Informed Features
# ================================================================
# STEP 66 - MODEL 1 V14: PHYSICS-INFORMED FEATURES
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pvlib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 70)
print("STEP 66 - MODEL 1 V14: PHYSICS-INFORMED FEATURES")
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
# 3. LOAD DATA
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
# 4. BASE FEATURES
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
# 5. BUILD PHYSICS-INFORMED FEATURES
# ----------------------------------------------------------------

print("\nCreating physics-informed features...")


# ------------------------------------------------
# 5.1 Solar elevation
# ------------------------------------------------

history_df["solar_elevation_deg"] = (
        90.0 - history_df["solar_zenith_angle"]
)


# ------------------------------------------------
# 5.2 Cosine of solar zenith
# ------------------------------------------------
# Represents the solar elevation geometry.
# Negative values at night are clipped to zero.

zenith_rad = np.deg2rad(history_df["solar_zenith_angle"])

history_df["sun_cos_zenith"] = np.cos(zenith_rad).clip(lower=0)


# ------------------------------------------------
# 5.3 Daylight flag
# ------------------------------------------------

history_df["daylight_flag"] = (
        history_df["solar_zenith_angle"] < 90
).astype(int)


# ------------------------------------------------
# 5.4 Relative optical air mass
# ------------------------------------------------
# Air mass is physically meaningful only when the sun
# is above the horizon.
#
# We use Kasten-Young model through pvlib.

zenith_for_airmass = history_df["solar_zenith_angle"].clip(
    lower=0,
    upper=89.9
)

relative_airmass = pvlib.atmosphere.get_relative_airmass(
    zenith_for_airmass,
    model="kastenyoung1989"
)

# Night-time airmass is not physically meaningful,
# so set it to zero for the ML feature.

relative_airmass = pd.Series(
    relative_airmass,
    index=history_df.index
)

relative_airmass.loc[history_df["daylight_flag"] == 0] = 0

history_df["relative_airmass"] = relative_airmass


# ------------------------------------------------
# 5.5 Pressure-corrected air mass
# ------------------------------------------------
# Approximation:
# absolute_airmass ≈ relative_airmass × surface_pressure / 1013.25

history_df["pressure_corrected_airmass"] = (
        history_df["relative_airmass"]
        * history_df["PS"]
        / 1013.25
)


# ------------------------------------------------
# 5.6 Module-to-ambient temperature difference
# ------------------------------------------------
# Captures thermal operating condition of the PV module.

history_df["module_ambient_delta"] = (
        history_df["module_temp__5063"]
        - history_df["ambient_temp__5062"]
)


# ------------------------------------------------
# 5.7 Irradiance × module-temperature interaction
# ------------------------------------------------
# Instead of assuming a particular PV temperature
# coefficient, we let the ML model learn the interaction.

history_df["poa_module_temp_interaction"] = (
        history_df["poa_irradiance__5061"]
        * history_df["module_temp__5063"]
)


# ------------------------------------------------
# 5.8 Irradiance × ambient-temperature interaction
# ------------------------------------------------

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


print(f"\nPhysics-informed features added: {len(physics_features)}")


# ----------------------------------------------------------------
# 6. MERGE FEATURES INTO FORECAST DATASET
# ----------------------------------------------------------------
# IMPORTANT:
# Features are created from the full historical hourly dataset
# and then merged on timestamp.

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
# 7. FINAL FEATURE LIST
# ----------------------------------------------------------------

v14_features = base_features + physics_features

print(f"\nBase features     : {len(base_features)}")
print(f"Physics features  : {len(physics_features)}")
print(f"Total V14 features: {len(v14_features)}")


# ----------------------------------------------------------------
# 8. CHECK MISSING VALUES
# ----------------------------------------------------------------

missing_physics = forecast_df[physics_features].isna().sum()

print("\nMissing physics-informed values:")
print(missing_physics)

total_missing = missing_physics.sum()

if total_missing > 0:
    print(
        f"\nWARNING: Total missing physics feature values = "
        f"{total_missing}"
    )
else:
    print("\nAll physics-informed features are complete.")


# ----------------------------------------------------------------
# 9. VALIDATION HELPER
# ----------------------------------------------------------------

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    r2 = r2_score(y_val, predictions)

    return mae, rmse, r2


# ----------------------------------------------------------------
# 10. ROLLING VALIDATION
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
    # TRAIN / VALIDATION MASKS
    # ------------------------------------------------------------

    train_mask = (
            (forecast_df["timestamp"] >= fold["train_start"])
            & (forecast_df["timestamp"] <= fold["train_end"])
    )

    val_mask = (
            (forecast_df["timestamp"] >= fold["val_start"])
            & (forecast_df["timestamp"] <= fold["val_end"])
    )


    train_data = forecast_df.loc[
        train_mask
        & forecast_df[target_col].notna()
        ].copy()

    val_data = forecast_df.loc[
        val_mask
        & forecast_df[target_col].notna()
        ].copy()


    X_train = train_data[v14_features]
    y_train = train_data[target_col]

    X_val = val_data[v14_features]
    y_val = val_data[target_col]


    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_val.shape}")


    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = lgb.LGBMRegressor(
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


    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=True
            )
        ],
    )


    # ------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------

    mae, rmse, r2 = evaluate_model(
        model,
        X_val,
        y_val
    )


    best_iteration = model.best_iteration_

    print("\nV14 Results:")
    print(f"Best iteration: {best_iteration}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")


    results.append({
        "fold": i,
        "validation_year": fold["year"],
        "train_rows": len(train_data),
        "validation_rows": len(val_data),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "best_iteration": best_iteration,
    })


# ----------------------------------------------------------------
# 11. SUMMARY
# ----------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL 1 V14 ROLLING VALIDATION SUMMARY")
print("=" * 70)

print(results_df.to_string(index=False))


avg_mae = results_df["mae"].mean()
avg_rmse = results_df["rmse"].mean()
avg_r2 = results_df["r2"].mean()

print("\nAverage V14 performance:")
print(f"MAE : {avg_mae:.4f}")
print(f"RMSE: {avg_rmse:.4f}")
print(f"R²  : {avg_r2:.4f}")


# ----------------------------------------------------------------
# 12. REFERENCE RESULTS
# ----------------------------------------------------------------

print("\nReference - V6:")
print("MAE : 14.7334")
print("RMSE: 30.0998")
print("R²  : 0.9097")

print("\nReference - V10:")
print("MAE : 14.3502")
print("RMSE: 29.9448")
print("R²  : 0.9107")

print("\nReference - V13:")
print("MAE : 14.3788")
print("RMSE: 29.9159")
print("R²  : 0.9109")


# ----------------------------------------------------------------
# 13. SAVE VALIDATION RESULTS
# ----------------------------------------------------------------

OUTPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V14_Physics_Informed_Validation.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nValidation results saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("V14 COMPLETE")
print("=" * 70)
