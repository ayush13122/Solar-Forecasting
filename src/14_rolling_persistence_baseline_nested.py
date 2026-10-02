from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT PATH
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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
# LOAD FORECASTING DATA
# ============================================================

forecast_df = pd.read_csv(forecast_path)

forecast_df["timestamp"] = pd.to_datetime(
    forecast_df["timestamp"]
)

forecast_df["target_timestamp"] = pd.to_datetime(
    forecast_df["target_timestamp"]
)


# ============================================================
# LOAD ORIGINAL DATA FOR CURRENT AC POWER
# ============================================================

ac_df = pd.read_csv(preprocessed_path)

ac_df["timestamp"] = pd.to_datetime(
    ac_df["timestamp"]
)

ac_df = ac_df[
    [
        "timestamp",
        "ac_power__5069"
    ]
].copy()


# ============================================================
# MERGE CURRENT AC WITH FORECAST DATA
# ============================================================

forecast_df = forecast_df.merge(
    ac_df,
    on="timestamp",
    how="left"
)


# print("=" * 70)
# print("STEP 50 - ROLLING PERSISTENCE BASELINE")
# print("=" * 70)


# ============================================================
# ROLLING VALIDATION FOLDS
# ============================================================

folds = [
    ("Fold 1", 2013, 2014),
    ("Fold 2", 2014, 2015),
    ("Fold 3", 2015, 2016)
]


results = []


# ============================================================
# RUN FOLDS
# ============================================================

for fold_name, train_end_year, val_year in folds:

    # print("\n" + "-" * 70)
    # print(
    #     f"{fold_name}: "
    #     f"TRAIN 2010-{train_end_year} → VALIDATE {val_year}"
    # )
    # print("-" * 70)

    val_start = pd.Timestamp(
        f"{val_year}-01-01"
    )

    val_end = pd.Timestamp(
        f"{val_year}-12-31 23:59:59"
    )

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    val_df = forecast_df[
        (forecast_df["target_timestamp"] >= val_start) &
        (forecast_df["target_timestamp"] <= val_end)
        ].copy()

    # print("Validation rows:", len(val_df))


    # --------------------------------------------------------
    # PERSISTENCE PREDICTION
    # --------------------------------------------------------

    baseline_pred = val_df[
        "ac_power__5069"
    ].values

    actual = val_df[
        "target_ac_power"
    ].values


    # --------------------------------------------------------
    # REMOVE INVALID CURRENT AC VALUES
    # --------------------------------------------------------

    valid = (
            ~pd.isna(baseline_pred)
            &
            ~pd.isna(actual)
    )

    baseline_pred = baseline_pred[valid]
    actual = actual[valid]


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        baseline_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            baseline_pred
        )
    )

    r2 = r2_score(
        actual,
        baseline_pred
    )


    # print("\nPersistence results:")
    # print("MAE :", round(mae, 4))
    # print("RMSE:", round(rmse, 4))
    # print("R²  :", round(r2, 4))


    results.append({
        "fold": fold_name,
        "validation_year": val_year,
        "validation_rows": len(actual),
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)


# print("\n" + "=" * 70)
# print("ROLLING PERSISTENCE SUMMARY")
# print("=" * 70)
#
# print(
#     results_df.to_string(index=False)
# )


# ============================================================
# AVERAGE
# ============================================================

# print("\nAverage persistence performance:")
#
# print(
#     "MAE :",
#     round(results_df["mae"].mean(), 4)
# )
#
# print(
#     "RMSE:",
#     round(results_df["rmse"].mean(), 4)
# )
#
# print(
#     "R²  :",
#     round(results_df["r2"].mean(), 4)
# )

