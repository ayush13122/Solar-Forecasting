# ================================================================
# STEP 75 - FINAL MODEL COMPARISON
# ================================================================
#
# Final confirmatory comparison on 2017-2018:
#
# 1. Persistence
# 2. Model 1 - V15 Two-Stage
# 3. Model 1 - V18 Recency Weighted
# 4. Model 2 - Intraday LightGBM
#
# NOTE:
# Model 2 result below is the previously recorded test result.
# It is not retrained in this script.
# ================================================================

from pathlib import Path
import pandas as pd


print("=" * 70)
print("STEP 75 - FINAL MODEL COMPARISON")
print("=" * 70)


# ----------------------------------------------------------------
# 1. FIND PROJECT ROOT
# ----------------------------------------------------------------

current_path = Path(__file__).resolve()

PROJECT_ROOT = None

for parent in current_path.parents:

    if (
            (parent / ".venv").exists()
            and
            (
                    parent
                    / "data"
                    / "processed"
                    / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
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
                    / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
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
# 2. RESULT FILES
# ----------------------------------------------------------------

PROCESSED_DIR = (
        PROJECT_ROOT
        / "data"
        / "processed"
)


V15_FILE = (
        PROCESSED_DIR
        / "Model1_V15_Final_Test_Results.csv"
)


V18_FILE = (
        PROCESSED_DIR
        / "Model1_V18_Final_Test_Results.csv"
)


FINAL_OUTPUT = (
        PROCESSED_DIR
        / "Final_Model_Comparison_2017_2018.csv"
)


# ----------------------------------------------------------------
# 3. LOAD V15
# ----------------------------------------------------------------

if not V15_FILE.exists():

    raise FileNotFoundError(
        f"V15 result file not found:\n{V15_FILE}"
    )


v15_df = pd.read_csv(
    V15_FILE
)


# V15 row
v15_row = v15_df[
    v15_df["model"]
    == "V15 Two-Stage"
    ]


if len(v15_row) == 0:

    raise ValueError(
        "V15 Two-Stage row not found."
    )


v15_row = v15_row.iloc[0]


# ----------------------------------------------------------------
# 4. LOAD V18
# ----------------------------------------------------------------

if not V18_FILE.exists():

    raise FileNotFoundError(
        f"V18 result file not found:\n{V18_FILE}"
    )


v18_df = pd.read_csv(
    V18_FILE
)


v18_row = v18_df[
    v18_df["model"]
    == "V18 Two-Stage Recency Weighted"
    ]


if len(v18_row) == 0:

    raise ValueError(
        "V18 result row not found."
    )


v18_row = v18_row.iloc[0]


# ----------------------------------------------------------------
# 5. EXTRACT PERSISTENCE
# ----------------------------------------------------------------
# V15/V18 both contain Persistence.
# Verify they agree.

v15_persistence = v15_df[
    v15_df["model"]
    == "Persistence"
    ]


v18_persistence = v18_df[
    v18_df["model"]
    == "Persistence"
    ]


if len(v15_persistence) == 0:

    raise ValueError(
        "Persistence row not found in V15 results."
    )


if len(v18_persistence) == 0:

    raise ValueError(
        "Persistence row not found in V18 results."
    )


v15_persistence = (
    v15_persistence.iloc[0]
)

v18_persistence = (
    v18_persistence.iloc[0]
)


# Use V18 persistence if available
persistence_mae = float(
    v18_persistence["mae"]
)

persistence_rmse = float(
    v18_persistence["rmse"]
)

persistence_r2 = float(
    v18_persistence["r2"]
)


# ----------------------------------------------------------------
# 6. MODEL 2 - PREVIOUSLY RECORDED TEST RESULT
# ----------------------------------------------------------------
#
# Model 2:
# Current AC + weather/solar features -> next-hour AC
#
# Previously recorded:
# MAE  = 12.2133
# RMSE = 24.3949
# R²   = 0.9235
#
# This script does not retrain Model 2.

MODEL2_MAE = 12.2133
MODEL2_RMSE = 24.3949
MODEL2_R2 = 0.9235


# ----------------------------------------------------------------
# 7. BUILD FINAL TABLE
# ----------------------------------------------------------------

comparison_df = pd.DataFrame({

    "model": [

        "Persistence",

        "Model 1 - V15 Two-Stage",

        "Model 1 - V18 Recency Weighted",

        "Model 2 - Intraday LightGBM",
    ],

    "mae": [

        persistence_mae,

        float(v15_row["mae"]),

        float(v18_row["mae"]),

        MODEL2_MAE,
    ],

    "rmse": [

        persistence_rmse,

        float(v15_row["rmse"]),

        float(v18_row["rmse"]),

        MODEL2_RMSE,
    ],

    "r2": [

        persistence_r2,

        float(v15_row["r2"]),

        float(v18_row["r2"]),

        MODEL2_R2,
    ],
})


# ----------------------------------------------------------------
# 8. DIFFERENCE VS PERSISTENCE
# ----------------------------------------------------------------

comparison_df[
    "mae_change_vs_persistence_pct"
] = (

        (
                comparison_df["mae"]
                - persistence_mae
        )
        / persistence_mae
        * 100

)


comparison_df[
    "rmse_change_vs_persistence_pct"
] = (

        (
                comparison_df["rmse"]
                - persistence_rmse
        )
        / persistence_rmse
        * 100

)


# ----------------------------------------------------------------
# 9. VALIDATION RESULTS REFERENCE
# ----------------------------------------------------------------

validation_reference_df = pd.DataFrame({

    "model": [

        "Model 1 - V15",

        "Model 1 - V18",
    ],

    "validation_mae": [

        14.1381,

        14.0155,
    ],

    "validation_rmse": [

        29.6294,

        29.4342,
    ],

    "validation_r2": [

        0.9125,

        0.9138,
    ],

})


# ----------------------------------------------------------------
# 10. PRINT FINAL TEST COMPARISON
# ----------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "FINAL 2017-2018 CONFIRMATORY TEST COMPARISON"
)

print(
    "=" * 70
)


print(
    comparison_df[
        [
            "model",
            "mae",
            "rmse",
            "r2",
            "mae_change_vs_persistence_pct",
            "rmse_change_vs_persistence_pct",
        ]
    ].to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 11. PRINT VALIDATION REFERENCE
# ----------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "MODEL 1 ROLLING VALIDATION REFERENCE"
)

print(
    "=" * 70
)


print(
    validation_reference_df.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 12. PRINT V15 VS V18 TEST DIFFERENCE
# ----------------------------------------------------------------

v15_test_mae = float(
    v15_row["mae"]
)

v18_test_mae = float(
    v18_row["mae"]
)


v15_test_rmse = float(
    v15_row["rmse"]
)

v18_test_rmse = float(
    v18_row["rmse"]
)


v15_test_r2 = float(
    v15_row["r2"]
)

v18_test_r2 = float(
    v18_row["r2"]
)


print(
    "\n" + "=" * 70
)

print(
    "V15 vs V18 ON 2017-2018"
)

print(
    "=" * 70
)


print(
    f"\nMAE : "
    f"V15 = {v15_test_mae:.4f} | "
    f"V18 = {v18_test_mae:.4f}"
)


print(
    f"RMSE: "
    f"V15 = {v15_test_rmse:.4f} | "
    f"V18 = {v18_test_rmse:.4f}"
)


print(
    f"R²  : "
    f"V15 = {v15_test_r2:.4f} | "
    f"V18 = {v18_test_r2:.4f}"
)


# ----------------------------------------------------------------
# 13. SAVE
# ----------------------------------------------------------------

comparison_df.to_csv(
    FINAL_OUTPUT,
    index=False
)


print(
    "\nFinal comparison saved to:"
)

print(
    FINAL_OUTPUT
)


print(
    "\n" + "=" * 70
)

print(
    "STEP 75 COMPLETE"
)

print(
    "=" * 70
)