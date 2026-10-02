from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT ROOT
# ============================================================

CURRENT_DIR = Path(__file__).resolve()

PROJECT_ROOT = None

for parent in [CURRENT_DIR] + list(CURRENT_DIR.parents):
    if (
            (parent / "data" / "processed").exists()
            and (parent / "src").exists()
    ):
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError("Project root could not be detected.")

print("=" * 70)
print("MODEL 2 EXPLAINABILITY")
print("=" * 70)
print(f"Project root: {PROJECT_ROOT}")


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model2_Intraday_NextHour.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "explainability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input file : {INPUT_PATH}")
print(f"Output dir : {OUTPUT_DIR}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])

print("\nDataset shape:", df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET = "target_ac_power"

if TARGET not in df.columns:
    raise KeyError(f"Target column '{TARGET}' not found.")


# ============================================================
# FEATURE COLUMNS
# ============================================================

EXCLUDE_COLUMNS = {
    "timestamp",
    "target_timestamp",
    TARGET
}

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in EXCLUDE_COLUMNS
]

print("\nNumber of features:", len(FEATURE_COLUMNS))

print("\nFeatures:")
for i, col in enumerate(FEATURE_COLUMNS, start=1):
    print(f"{i:02d}. {col}")


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_mask = df["target_timestamp"].dt.year <= 2015
val_mask = df["target_timestamp"].dt.year == 2016
test_mask = df["target_timestamp"].dt.year >= 2017

train_df = df.loc[train_mask].copy()
val_df = df.loc[val_mask].copy()
test_df = df.loc[test_mask].copy()

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# ============================================================
# X / y
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET]

X_val = val_df[FEATURE_COLUMNS]
y_val = val_df[TARGET]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET]


# ============================================================
# MODEL
# ============================================================

print("\nTraining LightGBM Model 2...")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=687,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse"
)

print("Model training completed.")


# ============================================================
# FEATURE IMPORTANCE — GAIN
# ============================================================

importance_gain = model.booster_.feature_importance(
    importance_type="gain"
)

importance_split = model.booster_.feature_importance(
    importance_type="split"
)

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "gain_importance": importance_gain,
    "split_importance": importance_split
})

importance_df["gain_percentage"] = (
        importance_df["gain_importance"]
        / importance_df["gain_importance"].sum()
        * 100
)

importance_df = importance_df.sort_values(
    "gain_importance",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "model2_feature_importance.csv",
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP FEATURES — GAIN IMPORTANCE")
print("=" * 70)

print(
    importance_df[
        ["feature", "gain_percentage"]
    ].head(15).to_string(index=False)
)


# ============================================================
# GAIN IMPORTANCE PLOT
# ============================================================

top_gain = importance_df.head(15).sort_values(
    "gain_percentage"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_gain["feature"],
    top_gain["gain_percentage"]
)

plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.title("Model 2 — LightGBM Feature Importance (Gain)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model2_feature_importance_gain.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SPLIT IMPORTANCE PLOT
# ============================================================

split_df = importance_df.sort_values(
    "split_importance",
    ascending=False
).head(15)

split_plot = split_df.sort_values(
    "split_importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    split_plot["feature"],
    split_plot["split_importance"]
)

plt.xlabel("Number of Splits")
plt.ylabel("Feature")
plt.title("Model 2 — LightGBM Feature Importance (Split)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model2_feature_importance_split.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP
# ============================================================

print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY")
print("=" * 70)

try:
    import shap
except ImportError:
    raise ImportError(
        "\nSHAP is not installed.\n"
        "Run:\n"
        "pip install shap"
    )


# ------------------------------------------------------------
# Use a sample of test data for SHAP
# ------------------------------------------------------------

SHAP_SAMPLE_SIZE = min(5000, len(X_test))

X_shap = X_test.sample(
    n=SHAP_SAMPLE_SIZE,
    random_state=42
)

print(f"SHAP sample size: {len(X_shap)}")


# ------------------------------------------------------------
# Tree Explainer
# ------------------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_shap)

shap_values = np.asarray(shap_values)

print("SHAP calculation completed.")


# ============================================================
# SHAP GLOBAL IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "mean_abs_shap": mean_abs_shap
})

shap_importance_df["importance_percentage"] = (
        shap_importance_df["mean_abs_shap"]
        / shap_importance_df["mean_abs_shap"].sum()
        * 100
)

shap_importance_df = shap_importance_df.sort_values(
    "mean_abs_shap",
    ascending=False
)

shap_importance_df.to_csv(
    OUTPUT_DIR / "model2_shap_global_importance.csv",
    index=False
)


# ============================================================
# PRINT SHAP TOP FEATURES
# ============================================================

print("\nTop SHAP features:")

print(
    shap_importance_df[
        ["feature", "importance_percentage"]
    ].head(15).to_string(index=False)
)


# ============================================================
# SHAP SUMMARY BAR
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    show=False,
    max_display=15
)

plt.title("Model 2 — SHAP Global Feature Importance")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model2_shap_summary_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP SUMMARY DOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    show=False,
    max_display=15
)

plt.title("Model 2 — SHAP Feature Impact")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model2_shap_summary_dot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE SHAP VALUES
# ============================================================

shap_values_df = pd.DataFrame(
    shap_values,
    columns=FEATURE_COLUMNS,
    index=X_shap.index
)

shap_values_df.to_csv(
    OUTPUT_DIR / "model2_shap_values_sample.csv",
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("EXPLAINABILITY COMPLETE")
print("=" * 70)

print(f"\nResults saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("1. model2_feature_importance.csv")
print("2. model2_feature_importance_gain.png")
print("3. model2_feature_importance_split.png")
print("4. model2_shap_global_importance.csv")
print("5. model2_shap_summary_bar.png")
print("6. model2_shap_summary_dot.png")
print("7. model2_shap_values_sample.csv")

print("\nDone.")