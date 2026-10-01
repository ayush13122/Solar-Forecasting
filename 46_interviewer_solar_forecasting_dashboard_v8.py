

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st


try:
    import lightgbm as lgb
except ImportError:
    st.error("LightGBM is required. Install it with: pip install lightgbm")
    st.stop()

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly is required. Install it with: pip install plotly")
    st.stop()

try:
    import pvlib
except ImportError:
    st.error("pvlib is required. Install it with: pip install pvlib")
    st.stop()

warnings.filterwarnings("ignore")




# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Power Forecasting & Plant Analytics",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT ROOT
# ============================================================

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = None

for parent in [CURRENT_DIR] + list(CURRENT_DIR.parents):
    if (
        (parent / "data" / "processed").exists()
        and (parent / "reports").exists()
    ):
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    st.error("Project root could not be detected.")
    st.stop()

# The dashboard file is expected to live in the project root.
# This diagnostic makes path issues obvious instead of failing silently.
st.sidebar.caption(f"Project root: {PROJECT_ROOT}")


# ============================================================
# PROJECT PATHS
# ============================================================


DATA_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_DIR = PROJECT_ROOT / "reports" / "performance_analytics"
EXPLAIN_DIR = PROJECT_ROOT / "reports" / "explainability"
EDA_DIR = PROJECT_ROOT / "reports" / "eda" / "deep_analysis"

MODEL1_FILE = DATA_DIR / "Model1_WeatherSolar_NextHour.csv"
MODEL2_FILE = DATA_DIR / "Model2_Intraday_NextHour.csv"
# Integrated Master Dataset
MASTER_FILE = DATA_DIR / "PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"

MODEL2_METRICS_FILE = REPORT_DIR / "model2_overall_metrics.csv"
MODEL2_PERF_FILE = REPORT_DIR / "model2_performance_test_predictions.csv"
MODEL2_HOURLY_FILE = REPORT_DIR / "model2_hourly_performance.csv"
MODEL2_MONTHLY_FILE = REPORT_DIR / "model2_monthly_performance.csv"
MODEL2_IRR_FILE = REPORT_DIR / "model2_irradiance_performance.csv"
MODEL2_CLOUD_FILE = REPORT_DIR / "model2_cloud_performance.csv"
MODEL2_YEARLY_FILE = REPORT_DIR / "model2_yearly_performance.csv"

SHAP_FILE = EXPLAIN_DIR / "model2_shap_global_importance.csv"
GAIN_FILE = EXPLAIN_DIR / "model2_feature_importance.csv"


# ============================================================
# EXACT MODEL FEATURE SET FROM THE PROJECT
# ============================================================

MODEL1_FEATURES = [
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

MODEL2_FEATURES = ["ac_power__5069"] + MODEL1_FEATURES
TARGET = "target_ac_power"


# ============================================================
# MODEL 1 RESEARCH CATALOG
# ============================================================

MODEL1_CATALOG = pd.DataFrame(
    [
        ["V1", "Baseline", "Original LightGBM with 26 features.", "Established the first Model 1 benchmark.", 37.0633, 0.8234, "Rejected vs persistence"],
        ["V2", "Time / Solar", "Added deterministic target-hour solar/time features.", "Tested explicit forecast-hour geometry.", 30.0928, np.nan, "Rejected"],
        ["V3", "Temporal Lag", "POA lag-1 + POA change.", "Added short-term irradiance dynamics.", 37.8146, 0.8162, "Rejected"],
        ["V4", "Regularization", "Regularized LightGBM.", "Tested whether overfitting was hurting transfer.", 37.2375, 0.8217, "Rejected"],
        ["V5", "Regime / Target", "Daylight specialist + normalized target.", "Separated night/low-power from active generation.", 38.3341, 0.8111, "Rejected"],
        ["V6", "Recency", "Recent-history LightGBM.", "Tested whether old history hurt later years.", 38.3097, 0.8113, "Rejected"],
        ["V7", "Target Transform", "Normalized-target version of V6.", "Tested target-scaling stability.", 38.3097, 0.8113, "Rejected / no change"],
        ["V8", "Algorithm", "XGBoost recent-history model.", "Tested another boosting family.", 40.7058, 0.7870, "Rejected"],
        ["V9", "Operating Regime", "Separate low-POA and active-POA models.", "Tested specialized regime models.", np.nan, np.nan, "Rejected"],
        ["V10", "Temporal Features", "Expanded lags, changes, rolling statistics.", "Captured recent weather/irradiance evolution.", 41.3294, 0.7804, "Rejected"],
        ["V11", "Clear Sky", "Clear-sky / clearness-index features.", "Added a physical irradiance normalization idea.", 40.6909, 0.7871, "Rejected"],
        ["V12", "Ensemble", "LightGBM + XGBoost weighted ensemble.", "Tested complementary model errors.", 40.4383, 0.7898, "Rejected"],
        ["V13", "Multivariate Temporal", "Expanded temporal history across multiple weather variables.", "Tested multivariate short-term dynamics.", np.nan, np.nan, "No confirmatory breakthrough"],
        ["V14", "Physics-Informed", "Solar elevation, air mass, temperature deltas/interactions.", "Added physically motivated relationships.", np.nan, np.nan, "Strong validation candidate"],
        ["V15", "Two-Stage", "Active/near-zero classifier + active-power regressor.", "Separated state classification from active generation.", 41.5746, 0.7778, "Rejected vs persistence"],
        ["V16", "Deep Learning", "24-hour LSTM sequence model.", "Tested whether sequence learning captured temporal context.", np.nan, np.nan, "Rejected"],
        ["V17", "Diagnosis", "Distribution-shift analysis; no new model.", "Explained the validation-to-confirmatory gap.", np.nan, np.nan, "Diagnostic — retained"],
        ["V18", "Recency Weighting", "Recency-weighted two-stage model.", "Explicitly prioritized newer patterns.", 41.9945, 0.7733, "Rejected vs persistence"],
        ["V19", "Target Transform", "Log1p active target.", "Tested skew-aware regression.", np.nan, np.nan, "Rejected"],
        ["V20", "Residual Calibration", "Residual correction model on top of V15-style output.", "Attempted systematic bias correction.", np.nan, np.nan, "Rejected"],
    ],
    columns=[
        "Version",
        "Theme",
        "What changed",
        "Why it was tried",
        "Confirmatory RMSE",
        "Confirmatory R2",
        "Decision",
    ],
)


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

required_files = [
    MODEL1_FILE,
    MODEL2_FILE,
    MASTER_FILE,
    MODEL2_METRICS_FILE,
    MODEL2_PERF_FILE,
    MODEL2_HOURLY_FILE,
    MODEL2_MONTHLY_FILE,
    MODEL2_IRR_FILE,
    MODEL2_CLOUD_FILE,
    MODEL2_YEARLY_FILE,
    SHAP_FILE,
    GAIN_FILE,
]

missing = [str(p) for p in required_files if not p.exists()]

if missing:
    st.error(
        "The dashboard cannot start because required project outputs are missing. "
        "This is a path/file issue, not a model issue."
    )
    for item in missing:
        st.code(item)
    st.stop()


# ============================================================
# DATA LOADING + NUMERIC SAFETY
# ============================================================

@st.cache_data
def load_project_data():
    model1 = pd.read_csv(MODEL1_FILE)
    model2 = pd.read_csv(MODEL2_FILE)
    master = pd.read_csv(MASTER_FILE)
    metrics = pd.read_csv(MODEL2_METRICS_FILE)
    perf = pd.read_csv(MODEL2_PERF_FILE)
    hourly = pd.read_csv(MODEL2_HOURLY_FILE)
    monthly = pd.read_csv(MODEL2_MONTHLY_FILE)
    irradiance = pd.read_csv(MODEL2_IRR_FILE)
    cloud = pd.read_csv(MODEL2_CLOUD_FILE)
    yearly = pd.read_csv(MODEL2_YEARLY_FILE)
    shap = pd.read_csv(SHAP_FILE)
    gain = pd.read_csv(GAIN_FILE)

    # Dates
    for frame in [model1, model2, master, perf]:
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            errors="coerce"
        )
        if "target_timestamp" in frame.columns:
            frame["target_timestamp"] = pd.to_datetime(
                frame["target_timestamp"],
                errors="coerce",
            )

    # NUMERIC SAFETY:
    # Ensure every LightGBM feature is genuinely numeric before prediction.
    for frame in [model1, model2, perf]:
        cols = [c for c in MODEL2_FEATURES if c in frame.columns]
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    target_frames = [model1, model2, perf]
    for frame in target_frames:
        if TARGET in frame.columns:
            frame[TARGET] = pd.to_numeric(
                frame[TARGET],
                errors="coerce",
            )

    for frame in [hourly, monthly, irradiance, cloud, yearly]:
        for col in frame.columns:
            if col not in ["irradiance_group", "cloud_group"]:
                frame[col] = pd.to_numeric(
                    frame[col],
                    errors="coerce",
                )

    for frame in [metrics, shap, gain]:
        for col in frame.columns:
            if col not in ["model", "feature"]:
                frame[col] = pd.to_numeric(
                    frame[col],
                    errors="coerce",
                )

    return (
        model1,
        model2,
        master,
        metrics,
        perf,
        hourly,
        monthly,
        irradiance,
        cloud,
        yearly,
        shap,
        gain,
    )


(
    model1_df,
    model2_df,
    master_df,
    metrics,
    perf,
    hourly,
    monthly,
    irradiance,
    cloud,
    yearly,
    shap_global,
    gain_global,
) = load_project_data()
# ============================================================
# DEEP EDA OUTPUTS
# ============================================================

def load_eda_csv(filename):
    path = EDA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


eda_hour_month = load_eda_csv("02_hour_month_ac_power_profile.csv")
eda_autocorr = load_eda_csv("03_ac_autocorrelation.csv")
eda_ramp = load_eda_csv("04_ramp_summary.csv")
eda_irradiance = load_eda_csv("05_irradiance_bin_analysis.csv")
eda_cloud = load_eda_csv("06_cloud_bin_analysis.csv")
eda_temperature = load_eda_csv("07_temperature_summary.csv")
eda_annual_energy = load_eda_csv("08_annual_gross_energy.csv")
eda_utilization = load_eda_csv("09_ac_capacity_utilization.csv")
eda_consistency = load_eda_csv("10_kwh_gross_ac_consistency.csv")
eda_daylight = load_eda_csv("11_daylight_vs_night.csv")
eda_sensor_corr = load_eda_csv("12_sensor_relationship_correlation.csv")
eda_yearly_quality = load_eda_csv("13_yearly_data_quality_patterns.csv")
eda_monthly_missing = load_eda_csv("14_monthly_missingness_patterns.csv")
eda_target_corr = load_eda_csv("15_all_target_correlations.csv")
eda_poa_regime = load_eda_csv("16_poa_operating_regime_summary.csv")

st.sidebar.success("Project data loaded successfully.")


# ============================================================
# EXACT MODEL 2 RE-CREATION — CACHED ONCE
# ============================================================

@st.cache_resource
def train_model2():
    train_end = pd.Timestamp("2015-12-31 23:59:59")
    val_start = pd.Timestamp("2016-01-01")
    val_end = pd.Timestamp("2016-12-31 23:59:59")

    train = model2_df[model2_df["timestamp"] <= train_end].copy()
    validation = model2_df[
        (model2_df["timestamp"] >= val_start)
        & (model2_df["timestamp"] <= val_end)
    ].copy()

    train = train.dropna(subset=MODEL2_FEATURES + [TARGET])
    validation = validation.dropna(subset=MODEL2_FEATURES + [TARGET])

    X_train = train[MODEL2_FEATURES].astype(float)
    y_train = train[TARGET].astype(float)

    X_val = validation[MODEL2_FEATURES].astype(float)
    y_val = validation[TARGET].astype(float)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    return model


# ============================================================
# COMMON VALUES
# ============================================================

model2_metric_row = metrics[
    metrics["model"].str.contains("Model 2", case=False, na=False)
].iloc[0]

baseline_metric_row = metrics[
    metrics["model"].str.contains("Persistence", case=False, na=False)
].iloc[0]

MODEL2_MAE = float(model2_metric_row["MAE"])
MODEL2_RMSE = float(model2_metric_row["RMSE"])
MODEL2_R2 = float(model2_metric_row["R2"])

BASELINE_MAE = float(baseline_metric_row["MAE"])
BASELINE_RMSE = float(baseline_metric_row["RMSE"])
BASELINE_R2 = float(baseline_metric_row["R2"])

MAE_IMPROVEMENT = (
    (BASELINE_MAE - MODEL2_MAE) / BASELINE_MAE * 100
)

RMSE_IMPROVEMENT = (
    (BASELINE_RMSE - MODEL2_RMSE) / BASELINE_RMSE * 100
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("☀️ Solar Power Forecasting & Plant Analytics")
st.caption(
    "Project presentation dashboard — from real PV plant data acquisition "
    "to preprocessing, Model 1 research, Model 2 forecasting, explainability and error analysis."
)

st.info(
    "Project scope: historical next-hour forecasting for NREL PVDAQ System 1433. "
    "Model 2 uses current-hour AC power plus environmental and temporal features."
)


# ============================================================
# NAVIGATION
# ============================================================

(
    tab_overview,
    tab_data,
    tab_eda,
    tab_design,
    tab_model1,
    tab_model2,
    tab_demo,
    tab_errors,
    tab_explain,
    tab_method,
) = st.tabs(
    [
        "Executive Overview",
        "Data Journey",
        "EDA Insights",
        "Forecast Design",
        "Model 1 Research",
        "Model 2 Results",
        "Forecast Demonstration",
        "Error Analysis",
        "Explainability",
        "Methodology & Scope",
    ]
)


# ============================================================
# 1. EXECUTIVE OVERVIEW
# ============================================================

with tab_overview:
    st.header("Executive Overview")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Model 2 MAE", f"{MODEL2_MAE:.2f} kW")
    k2.metric("Model 2 RMSE", f"{MODEL2_RMSE:.2f} kW")
    k3.metric("R²", f"{MODEL2_R2:.4f}")
    k4.metric("RMSE vs Persistence", f"-{RMSE_IMPROVEMENT:.2f}%")

    st.divider()

    st.subheader("The question the project answers")

    st.markdown(
        """
        **Given the information available at hour T, can we estimate the plant's
        AC power at T+1 — and can we explain where the forecast is reliable,
        where it struggles, and why?**
        """
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("The strongest quantitative result")

        with st.container(border=True):
            st.write(
                f"Model 2 achieved **R² = {MODEL2_R2:.4f}**, "
                f"**MAE = {MODEL2_MAE:.4f} kW**, and "
                f"**RMSE = {MODEL2_RMSE:.4f} kW** on the evaluated "
                f"2017–2018 confirmatory period."
            )
            st.caption(
                "R² is a variance-explained metric, not a per-record accuracy percentage."
            )

    with c2:
        st.subheader("The strongest research insight")

        with st.container(border=True):
            st.write(
                "Model 1 repeatedly improved rolling validation, but those gains "
                "did not reliably transfer to the later confirmatory period."
            )
            st.caption(
                "This validation-to-confirmatory gap became a central modeling lesson."
            )

    st.divider()

    st.subheader("Model comparison")

    comparison = pd.DataFrame(
        {
            "Model": ["Persistence", "Model 2"],
            "MAE": [BASELINE_MAE, MODEL2_MAE],
            "RMSE": [BASELINE_RMSE, MODEL2_RMSE],
        }
    )

    fig = px.bar(
        comparison,
        x="Model",
        y=["MAE", "RMSE"],
        barmode="group",
        title="Next-Hour Forecast Error",
        labels={"value": "Error (kW)", "variable": "Metric"},
    )

    st.plotly_chart(fig, width="stretch")


# ============================================================
# 2. DATA JOURNEY
# ============================================================

with tab_data:
    st.header("Data Journey")
    st.caption("How the raw plant data became the ML-ready forecasting dataset.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("PVDAQ System", "1433")
    c2.metric("Raw Rows", "246,903")
    c3.metric("Raw Resolution", "15 min")
    c4.metric("Coverage", "2010–2018")

    st.divider()

    pipeline = pd.DataFrame(
        {
            "Stage": [
                "1. Raw PVDAQ",
                "2. Hourly aggregation",
                "3. NASA POWER integration",
                "4. ERA5 integration",
                "5. 900 hPa wind",
                "6. Solar position",
                "7. Audit + preprocessing",
                "8. Forecast pair creation",
            ],
            "Result": [
                "246,903 × 10",
                "61,726 hourly rows",
                "Temperature, RH, wind, pressure, precipitation, irradiance",
                "Cloud layers, gust, snowfall",
                "Wind speed + direction at 900 hPa",
                "Solar zenith + azimuth",
                "ML-ready rows and quality decisions",
                "56,275 valid one-hour pairs",
            ],
        }
    )

    st.dataframe(
        pipeline,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # ========================================================
    # KEY DATASETS
    # ========================================================

    st.subheader("Key Project Datasets")

    d1, d2 = st.columns(2)

    # --------------------------------------------------------
    # DATASET 1 — INTEGRATED MASTER
    # --------------------------------------------------------

    with d1:
        st.markdown("### 1. Integrated Master Dataset")

        st.write(
            "The final integrated hourly dataset combining real PVDAQ "
            "plant measurements with NASA POWER and ERA5 environmental data."
        )

        st.metric(
            "Rows × Columns",
            f"{master_df.shape[0]:,} × {master_df.shape[1]}"
        )

        st.caption(
            "PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
        )

        st.dataframe(
            master_df.head(10),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download Master Dataset",
            data=master_df.to_csv(index=False).encode("utf-8"),
            file_name="PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv",
            mime="text/csv",
            key="download_master_dataset",
        )

    # --------------------------------------------------------
    # DATASET 2 — MODEL 2 FORECASTING DATASET
    # --------------------------------------------------------

    with d2:
        st.markdown("### 2. Model 2 Forecasting Dataset")

        st.write(
            "The final forecasting dataset used by Model 2 for "
            "intraday next-hour AC power prediction."
        )

        st.metric(
            "Rows × Columns",
            f"{model2_df.shape[0]:,} × {model2_df.shape[1]}"
        )

        st.caption(
            "Model2_Intraday_NextHour.csv"
        )

        st.dataframe(
            model2_df.head(10),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download Model 2 Dataset",
            data=model2_df.to_csv(index=False).encode("utf-8"),
            file_name="Model2_Intraday_NextHour.csv",
            mime="text/csv",
            key="download_model2_dataset",
        )

    st.divider()

    st.subheader("Data Transformation Story")

    journey = pd.DataFrame(
        {
            "Stage": [
                "Integrated Master",
                "ML-ready preprocessing",
                "Model 1 forecasting dataset",
                "Model 2 forecasting dataset",
            ],
            "Dataset": [
                "PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv",
                "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv",
                "Model1_WeatherSolar_NextHour.csv",
                "Model2_Intraday_NextHour.csv",
            ],
            "Rows × Columns": [
                "61,726 × 25",
                "56,980 × 28",
                "56,275 × 29",
                "56,275 × 30",
            ],
            "Purpose": [
                "Integrated plant + environmental master",
                "Evidence-based preprocessing and ML-ready features",
                "Weather/solar-based T+1 forecasting",
                "Intraday T+1 forecasting with current AC power",
            ],
        }
    )

    st.dataframe(
        journey,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Key data-quality decisions")

    dq = pd.DataFrame(
        {
            "Issue": [
                "Missing AC target",
                "Missing core sensors",
                "DC voltage",
                "PR",
                "Negative AC",
                "Negative POA",
                "Duplicate timestamps",
            ],
            "Observed evidence": [
                "4,741 missing hours; 702 blocks",
                "1,659 each for ambient/module/POA; exact overlap",
                "75.32% missing",
                "53.69% missing + extreme values",
                "1,267 negative observations initially",
                "Negative POA observations included one residual value after contextual handling",
                "0 duplicates",
            ],
            "Decision": [
                "Exclude target-missing rows from supervised ML",
                "Exclude rows where AC exists but all three core sensors are missing",
                "Excluded from primary ML feature set",
                "Excluded from primary ML feature set",
                "Locked to 0 in final ML data; anomaly evidence retained separately",
                "Locked residual negative value to 0 in final ML data",
                "No action required",
            ],
        }
    )

    st.dataframe(
        dq,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "The master dataset was preserved; preprocessing produced an ML-ready copy for traceability."
    )


## ============================================================
# 3. EDA
# ============================================================

with tab_eda:
    st.header("EDA Insights")
    st.caption(
        "The EDA stage was used to understand plant behaviour, temporal patterns, "
        "physical relationships, data quality and the conditions that make forecasting difficult."
    )

    # ========================================================
    # 1. EXECUTIVE EDA INSIGHT CARDS
    # ========================================================

    st.subheader("Key Findings from Deep EDA")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "1-hour AC correlation",
        "0.915",
        help="Correlation between AC power and the previous hourly AC power."
    )

    c2.metric(
        "24-hour AC correlation",
        "0.838",
        help="Correlation between AC power and the same-hour previous-day AC power."
    )

    c3.metric(
        "Mean absolute hourly ramp",
        "22.49 kW",
        help="Average absolute hour-to-hour change in AC power."
    )

    c4.metric(
        "99th percentile ramp",
        "142 kW",
        help="Large but relatively uncommon 1-hour AC power changes."
    )

    st.divider()

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "POA > 1000 W/m²",
        "314.47 kW",
        help="Mean AC power in the >1000 W/m² POA regime."
    )

    c6.metric(
        "POA 0–100 W/m²",
        "7.05 kW",
        help="Mean AC power in the 0–100 W/m² POA regime."
    )

    c7.metric(
        "Cloud 0–20%",
        "102.23 kW",
        help="Mean AC power under low cloud-cover conditions."
    )

    c8.metric(
        "Cloud 80–100%",
        "37.00 kW",
        help="Mean AC power under high cloud-cover conditions."
    )

    st.divider()

    # ========================================================
    # 2. WHAT EDA ACTUALLY TOLD US
    # ========================================================

    st.subheader("What Did the EDA Tell Us?")

    e1, e2 = st.columns(2)

    with e1:
        with st.container(border=True):
            st.markdown("### Strong short-term dependence")
            st.write(
                "AC power has a 1-hour correlation of approximately 0.915, "
                "showing strong short-term temporal dependence. The 24-hour "
                "correlation is also high at approximately 0.838, confirming "
                "strong daily generation structure."
            )

    with e2:
        with st.container(border=True):
            st.markdown("### Forecasting is hardest during active generation")
            st.write(
                "Low-irradiance periods have very low generation and comparatively "
                "small absolute errors. As irradiance increases, power output and "
                "the magnitude of forecasting errors become much larger."
            )

    e3, e4 = st.columns(2)

    with e3:
        with st.container(border=True):
            st.markdown("### Weather conditions matter")
            st.write(
                "Mean generation decreases steadily across higher cloud-cover "
                "regimes. This supports including cloud and atmospheric variables "
                "in the forecasting feature set."
            )

    with e4:
        with st.container(border=True):
            st.markdown("### The plant has meaningful ramp behaviour")
            st.write(
                "The mean absolute hourly AC change is approximately 22.49 kW, "
                "while the 95th percentile absolute change is about 94 kW and "
                "the 99th percentile is about 142 kW. This indicates that rapid "
                "generation changes are an important forecasting challenge."
            )

    st.divider()

    # ========================================================
    # 3. HOUR × MONTH HEATMAP
    # ========================================================

    if eda_hour_month is not None:
        st.subheader("Generation Structure Across Time")

        heatmap_data = eda_hour_month.copy()

        if "hour" in heatmap_data.columns:
            heatmap_data = heatmap_data.set_index("hour")

        # Handle either numeric month columns or strings.
        heatmap_data.columns = [
            str(col) for col in heatmap_data.columns
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                hovertemplate=(
                    "Month: %{x}<br>"
                    "Hour: %{y}<br>"
                    "Mean AC Power: %{z:.2f} kW"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Mean AC Power by Hour and Month",
            xaxis_title="Month",
            yaxis_title="Hour of Day",
            height=600,
        )

        st.plotly_chart(fig, width="stretch")

        st.caption(
            "The heatmap makes the daily solar-generation cycle and its seasonal "
            "variation visible in one view."
        )

    # ========================================================
    # 4. AUTOCORRELATION + RAMPS
    # ========================================================

    a1, a2 = st.columns(2)

    with a1:
        if eda_autocorr is not None:
            fig = px.line(
                eda_autocorr,
                x="lag_hours",
                y="correlation",
                markers=True,
                title="AC Power Autocorrelation",
                labels={
                    "lag_hours": "Lag (hours)",
                    "correlation": "Correlation",
                },
            )

            fig.add_hline(y=0, line_dash="dash")

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.caption(
                "The strong lag-1 and lag-24 relationships justify looking at "
                "recent plant state and daily periodicity during forecasting design."
            )

    with a2:
        if eda_ramp is not None:
            ramp_values = eda_ramp.set_index("metric")["value"]

            ramp_df = pd.DataFrame(
                {
                    "Metric": [
                        "Mean absolute change",
                        "95th percentile absolute change",
                        "99th percentile absolute change",
                        "Maximum absolute change",
                    ],
                    "Value": [
                        ramp_values.get("mean_absolute_change", np.nan),
                        ramp_values.get(
                            "95th_percentile_absolute_change",
                            np.nan,
                        ),
                        ramp_values.get(
                            "99th_percentile_absolute_change",
                            np.nan,
                        ),
                        max(
                            abs(ramp_values.get("min_change", np.nan)),
                            abs(ramp_values.get("max_change", np.nan)),
                        ),
                    ],
                }
            )

            fig = px.bar(
                ramp_df,
                x="Metric",
                y="Value",
                title="Hourly AC Ramp Behaviour",
                labels={"Value": "Power Change (kW)"},
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.caption(
                "Large ramps matter because a next-hour forecast must react to "
                "rapid changes in generation."
            )

    st.divider()

    # ========================================================
    # 5. POA IRRADIANCE RESPONSE
    # ========================================================

    st.subheader("How AC Power Responds to Irradiance")

    if eda_irradiance is not None:
        fig = px.bar(
            eda_irradiance,
            x="poa_bin",
            y="mean",
            error_y="std",
            title="Mean AC Power Across POA Irradiance Regimes",
            labels={
                "poa_bin": "POA Irradiance Regime",
                "mean": "Mean AC Power (kW)",
            },
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        irr_display = eda_irradiance.rename(
            columns={
                "poa_bin": "POA Regime",
                "count": "Records",
                "mean": "Mean AC Power (kW)",
                "median": "Median AC Power (kW)",
                "std": "Std Dev",
                "min": "Minimum",
                "max": "Maximum",
            }
        )

        st.dataframe(
            irr_display,
            width="stretch",
            hide_index=True,
        )

        st.info(
            "Generation rises sharply with increasing POA. This is one of the strongest "
            "physical relationships visible in the EDA and later reflected in model explainability."
        )

    # ========================================================
    # 6. CLOUD COVER
    # ========================================================

    st.subheader("Cloud-Cover Regimes")

    if eda_cloud is not None:
        fig = px.bar(
            eda_cloud,
            x="cloud_bin",
            y="mean_ac_power",
            title="Mean AC Power by Cloud-Cover Regime",
            labels={
                "cloud_bin": "Cloud Cover",
                "mean_ac_power": "Mean AC Power (kW)",
            },
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        cloud_display = eda_cloud.rename(
            columns={
                "cloud_bin": "Cloud Cover",
                "mean_ac_power": "Mean AC Power (kW)",
                "median_ac_power": "Median AC Power (kW)",
                "mean_poa": "Mean POA (W/m²)",
                "count": "Records",
            }
        )

        st.dataframe(
            cloud_display,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Low cloud cover corresponds to substantially higher average generation, "
            "while high cloud cover is associated with lower irradiance and lower output."
        )

    # ========================================================
    # 7. TEMPERATURE
    # ========================================================

    st.subheader("Temperature Behaviour")

    if eda_temperature is not None:
        temp_display = eda_temperature.copy()

        st.dataframe(
            temp_display,
            width="stretch",
            hide_index=True,
        )

        st.write(
            "The module temperature is generally different from ambient temperature, "
            "and that relationship changes with solar loading. This supports keeping "
            "both ambient and module temperature in the feature set."
        )

    # ========================================================
    # 8. ENERGY + YEARLY CONTEXT
    # ========================================================

    st.divider()

    st.subheader("Energy Generation Context")

    if eda_annual_energy is not None:
        yearly_energy = eda_annual_energy.copy()

        # Explicitly label this as the recorded gross-energy field.
        yearly_energy.columns = [
            "Year",
            "Recorded Gross Energy",
        ]

        fig = px.bar(
            yearly_energy,
            x="Year",
            y="Recorded Gross Energy",
            title="Annual Recorded Gross Energy",
            labels={
                "Recorded Gross Energy": "Recorded Gross Energy",
            },
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.dataframe(
            yearly_energy,
            width="stretch",
            hide_index=True,
        )

        st.warning(
            "2010 and 2018 are partial years in the downloaded dataset, "
            "so their annual totals should not be compared directly with full years."
        )

    # ========================================================
    # 9. DAYLIGHT VS NIGHT
    # ========================================================

    d1, d2 = st.columns(2)

    with d1:
        if eda_daylight is not None:
            daylight_display = eda_daylight.copy()

            if "period" in daylight_display.columns:
                daylight_display = daylight_display.rename(
                    columns={
                        "period": "Period",
                        "count": "Records",
                        "mean": "Mean AC Power (kW)",
                        "median": "Median AC Power (kW)",
                        "std": "Std Dev",
                        "min": "Minimum",
                        "max": "Maximum",
                    }
                )

            st.subheader("Daylight vs Night")

            st.dataframe(
                daylight_display,
                width="stretch",
                hide_index=True,
            )

    with d2:
        if eda_poa_regime is not None:
            st.subheader("Low vs Active POA Regimes")

            regime_display = eda_poa_regime.rename(
                columns={
                    "regime": "Regime",
                    "count": "Records",
                    "mean": "Mean AC Power (kW)",
                    "median": "Median AC Power (kW)",
                    "std": "Std Dev",
                    "min": "Minimum",
                    "max": "Maximum",
                    "percentage": "Share (%)",
                }
            )

            st.dataframe(
                regime_display,
                width="stretch",
                hide_index=True,
            )

    # ========================================================
    # 10. CAPACITY UTILIZATION
    # ========================================================

    if eda_utilization is not None:
        st.subheader("AC Output Relative to Nominal DC Capacity")

        utilization_map = dict(
            zip(
                eda_utilization["metric"],
                eda_utilization["value"],
            )
        )

        u1, u2, u3, u4 = st.columns(4)

        u1.metric(
            "Mean Utilization",
            f"{utilization_map.get('mean_utilization', np.nan) * 100:.2f}%",
        )

        u2.metric(
            "Median Utilization",
            f"{utilization_map.get('median_utilization', np.nan) * 100:.2f}%",
        )

        u3.metric(
            "95th Percentile",
            f"{utilization_map.get('95th_percentile', np.nan) * 100:.2f}%",
        )

        u4.metric(
            "99th Percentile",
            f"{utilization_map.get('99th_percentile', np.nan) * 100:.2f}%",
        )

        st.caption(
            "This is a utilization-style diagnostic relative to the nominal DC capacity "
            "used in the project, not an inverter-clipping diagnosis."
        )

    # ========================================================
    # 11. kWh vs AC CONSISTENCY
    # ========================================================

    if eda_consistency is not None:
        st.subheader("Energy Consistency Check")

        consistency_map = dict(
            zip(
                eda_consistency["metric"],
                eda_consistency["value"],
            )
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Valid Rows",
            f"{int(consistency_map.get('valid_rows', 0)):,}",
        )

        c2.metric(
            "Median Energy / AC Ratio",
            f"{consistency_map.get('ratio_median', np.nan):.3f}",
        )

        c3.metric(
            "90% Range Upper",
            f"{consistency_map.get('ratio_90th_percentile', np.nan):.3f}",
        )

        st.caption(
            "This is a consistency diagnostic between the recorded gross-energy field "
            "and hourly AC power. It is not a model feature or a proof of sensor identity."
        )

    # ========================================================
    # 12. DATA-QUALITY PATTERNS
    # ========================================================

    st.divider()

    st.subheader("Data-Quality Patterns That Influenced Modeling")

    if eda_yearly_quality is not None:
        st.write("Year-level quality patterns")

        st.dataframe(
            eda_yearly_quality,
            width="stretch",
            hide_index=True,
        )

    if eda_monthly_missing is not None:
        st.write("Month-level missingness patterns")

        st.dataframe(
            eda_monthly_missing,
            width="stretch",
            hide_index=True,
        )

    st.info(
        "The key point is that missingness was not treated as harmless random noise. "
        "The project observed block gaps and therefore avoided blindly interpolating the target."
    )

    # ========================================================
    # 13. CORRELATION INSIGHTS
    # ========================================================

    st.subheader("Relationships With AC Power")

    if eda_target_corr is not None:
        corr_df = eda_target_corr.copy()

        # If the CSV saved the feature names as the first column/index,
        # normalize it for dashboard display.
        if "correlation_with_ac_power" in corr_df.columns:
            if corr_df.columns[0] not in [
                "correlation_with_ac_power",
                "absolute_correlation",
                "feature",
            ]:
                corr_df = corr_df.rename(
                    columns={corr_df.columns[0]: "feature"}
                )

        if "feature" not in corr_df.columns:
            corr_df = corr_df.reset_index().rename(
                columns={"index": "feature"}
            )

        corr_df = corr_df.head(15)

        fig = px.bar(
            corr_df.sort_values("correlation_with_ac_power"),
            x="correlation_with_ac_power",
            y="feature",
            orientation="h",
            title="Top Correlations With AC Power",
            labels={
                "correlation_with_ac_power": "Correlation",
                "feature": "Feature",
            },
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # ========================================================
    # 14. SENSOR RELATIONSHIPS
    # ========================================================

    if eda_sensor_corr is not None:
        st.subheader("Core Sensor Correlation Matrix")

        sensor_fig = go.Figure(
            data=go.Heatmap(
                z=eda_sensor_corr.values,
                x=eda_sensor_corr.columns,
                y=eda_sensor_corr.index,
                hovertemplate=(
                    "%{y} vs %{x}<br>"
                    "Correlation: %{z:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        sensor_fig.update_layout(
            title="Sensor Correlation Matrix",
            height=550,
        )

        st.plotly_chart(
            sensor_fig,
            width="stretch",
        )

    # ========================================================
    # 15. FINAL EDA TAKEAWAY
    # ========================================================

    st.divider()

    st.subheader("What EDA Changed in the Project")

    takeaway = pd.DataFrame(
        {
            "Observation": [
                "Strong 1-hour temporal dependence",
                "Strong daily periodicity",
                "Power rises with irradiance",
                "Generation decreases with cloud cover",
                "Large hourly ramps exist",
                "Missingness occurs in blocks",
                "Night and active generation behave differently",
                "Later forecasting errors concentrate in active generation",
            ],
            "Modeling implication": [
                "Current plant state can be valuable for T+1 intraday forecasting.",
                "Solar-position and cyclic temporal features are justified.",
                "POA should remain a core predictive feature.",
                "Cloud/environmental features are meaningful supporting inputs.",
                "A forecasting system must handle rapid generation transitions.",
                "Target imputation should be conservative.",
                "Regime-aware experiments are worth testing.",
                "Aggregate R² alone is not enough; regime-level error analysis is required.",
            ],
        }
    )

    st.dataframe(
        takeaway,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 4. FORECAST DESIGN
# ============================================================

with tab_design:
    st.header("Forecast Design")
    st.caption(
        "The two models use the same T → T+1 target but represent two different "
        "levels of information availability: weather/solar context versus current plant state."
    )

    st.subheader("What happens at forecasting time T?")

    flow = pd.DataFrame(
        {
            "Step": [
                "1. Observe hour T",
                "2. Build the feature vector",
                "3. Predict hour T+1",
                "4. Evaluate against the actual T+1 output",
            ],
            "Meaning": [
                "Use the information that is legitimately available at the selected hour.",
                "Combine plant/weather/solar features without using the future target value.",
                "Estimate the next-hour AC power of the same PV system.",
                "Compare the forecast with the recorded T+1 AC power when an evaluated pair exists.",
            ],
        }
    )

    st.dataframe(
        flow,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("What are the two models?")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("Model 1 — Weather / Solar")
            st.markdown("**26 features → T+1 AC power**")
            st.write(
                "Current AC power was deliberately excluded. The model relies on environmental, "
                "irradiance, solar-position and temporal information to estimate the next hour. "
                "This isolates how much predictive information can be obtained without direct "
                "knowledge of the plant's current electrical state."
            )
            st.caption(
                "Best suited conceptually to longer horizons only when the required future weather "
                "inputs are available as actual forecasts rather than historical observations."
            )

    with right:
        with st.container(border=True):
            st.subheader("Model 2 — Intraday")
            st.markdown("**27 features → T+1 AC power**")
            st.write(
                "The 26 Model 1 features plus current-hour AC power. The added plant-state signal "
                "lets the model respond to the system's present operating condition, making this "
                "an intraday T+1 forecasting formulation."
            )
            st.caption(
                "This is the stronger representation for short-horizon operating-state forecasting "
                "because current AC power is available at T."
            )

    st.divider()

    st.subheader("Why the target is always T+1")

    st.write(
        "Both models were evaluated against the same next-hour target so that the comparison stays "
        "clean. Model 1 asks how far weather/solar information alone can go; Model 2 asks how much "
        "additional predictive value is obtained when the current plant state is also known. "
        "This is a forecasting-design comparison, not a claim that both models solve every forecast horizon."
    )

    st.subheader("Which information is available at each stage?")

    availability = pd.DataFrame(
        {
            "Information at T": [
                "Current AC power",
                "Plant / weather sensor context",
                "Solar geometry",
                "Future weather forecast",
            ],
            "Model 1": [
                "Not used",
                "Used",
                "Used",
                "Not available in the current historical pipeline",
            ],
            "Model 2": [
                "Used",
                "Used",
                "Used",
                "Not required for current T+1 setup",
            ],
        }
    )

    st.dataframe(
        availability,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Why not call this a day-ahead forecast?")

    st.warning(
        "The historical NASA POWER and ERA5 inputs used in this project are observations/reanalysis. "
        "They are not operational future weather forecasts. A true day-ahead system would need future "
        "weather forecasts, forecast-derived irradiance/cloud information and a horizon-specific evaluation protocol."
    )

    st.subheader("Why LightGBM?")

    st.write(
        "The project used LightGBM because the data are structured/tabular, contain mixed environmental "
        "signals, nonlinear relationships and interactions, and the model supports efficient tree-based "
        "feature importance and SHAP explainability."
    )


# ============================================================
# 5. MODEL 1 RESEARCH
# ============================================================

with tab_model1:
    st.header("Model 1 Research Lab")
    st.caption(
        "Twenty variants are summarized as a research path so the dashboard remains readable."
    )

    st.subheader("What were we trying to solve?")

    st.write(
        "The research question was whether a weather/solar-only model could reliably "
        "predict next-hour plant output without using current AC power."
    )

    chart = MODEL1_CATALOG.dropna(subset=["Confirmatory RMSE"]).copy()

    fig = px.line(
        chart,
        x="Version",
        y="Confirmatory RMSE",
        markers=True,
        text="Confirmatory RMSE",
        title="Model 1 Confirmatory RMSE Across Tested Variants",
        labels={"Confirmatory RMSE": "Confirmatory RMSE (kW)"},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="top center")
    st.plotly_chart(fig, width="stretch")

    st.divider()

    selected_version = st.selectbox(
        "Inspect one experiment",
        MODEL1_CATALOG["Version"].tolist(),
        index=0,
    )

    selected = MODEL1_CATALOG[
        MODEL1_CATALOG["Version"] == selected_version
    ].iloc[0]

    st.subheader(
        f"{selected['Version']} — {selected['Theme']}"
    )

    detail = pd.DataFrame(
        {
            "Field": [
                "What changed",
                "Why it was tried",
                "Confirmatory RMSE",
                "Confirmatory R²",
                "Decision",
            ],
            "Details": [
                selected["What changed"],
                selected["Why it was tried"],
                "—"
                if pd.isna(selected["Confirmatory RMSE"])
                else f"{selected['Confirmatory RMSE']:.4f} kW",
                "—"
                if pd.isna(selected["Confirmatory R2"])
                else f"{selected['Confirmatory R2']:.4f}",
                selected["Decision"],
            ],
        }
    )

    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("The main lesson from V1–V20")

    st.info(
        "Several Model 1 variants improved rolling validation. The harder problem was "
        "transfer to the later confirmatory period. That gap became the main research insight, "
        "not something to hide."
    )

    # ========================================================
    # TOP-GRADE ENHANCEMENT 3 — VALIDATION → CONFIRMATORY GAP
    # ========================================================

    st.subheader("Validation → Confirmatory Transfer Diagnostic")

    # V15 is the documented two-stage Model 1 candidate for which both
    # rolling-validation and confirmatory-period RMSE are available.
    v15_transfer = pd.DataFrame(
        {
            "Evaluation Stage": [
                "Average rolling validation (2014–2016)",
                "Confirmatory period (2017–2018)",
            ],
            "RMSE (kW)": [
                29.6294,
                41.5746,
            ],
        }
    )

    transfer_fig = px.bar(
        v15_transfer,
        x="Evaluation Stage",
        y="RMSE (kW)",
        text="RMSE (kW)",
        title="V15 Two-Stage Model — Validation vs Confirmatory RMSE",
        labels={
            "Evaluation Stage": "",
            "RMSE (kW)": "RMSE (kW)",
        },
    )

    transfer_fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    transfer_fig.update_layout(
        yaxis_title="RMSE (kW)",
        height=460,
    )

    st.plotly_chart(
        transfer_fig,
        width="stretch",
    )

    validation_rmse = 29.6294
    confirmatory_rmse = 41.5746
    relative_gap = (
        (confirmatory_rmse - validation_rmse)
        / validation_rmse
        * 100
    )

    g1, g2, g3 = st.columns(3)

    g1.metric(
        "Rolling validation RMSE",
        f"{validation_rmse:.2f} kW",
    )

    g2.metric(
        "Confirmatory RMSE",
        f"{confirmatory_rmse:.2f} kW",
    )

    g3.metric(
        "Increase in RMSE",
        f"{relative_gap:.1f}%",
    )

    st.caption(
        "This diagnostic shows the observed transfer gap for V15. "
        "The confirmatory period was inspected during earlier Model 1 experimentation, "
        "so it is described as confirmatory rather than a pristine untouched test set."
    )

    st.dataframe(
        MODEL1_CATALOG,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 6. MODEL 2 RESULTS
# ============================================================

with tab_model2:
    st.header("Model 2 — Final Intraday Result")
    st.caption(
        "Evaluation period: 2017–2018 confirmatory period • 5,522 records"
    )

    a, b, c, d = st.columns(4)

    a.metric("MAE", f"{MODEL2_MAE:.4f} kW")
    b.metric("RMSE", f"{MODEL2_RMSE:.4f} kW")
    c.metric("R²", f"{MODEL2_R2:.4f}")
    d.metric(
        "MAE improvement vs persistence",
        f"{MAE_IMPROVEMENT:.2f}%",
    )

    st.divider()

    st.subheader("How should R² = 0.9235 be interpreted?")

    st.markdown(
        """
        **It does not mean 92.35% per-record accuracy.**

        R² describes how much of the variance in the evaluated target is explained by the model.

        The practical error measures are:
        """
    )

    metrics_table = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
                "R²",
                "Mean signed error",
                "Minimum error",
                "Maximum error",
            ],
            "Value": [
                f"{MODEL2_MAE:.4f} kW",
                f"{MODEL2_RMSE:.4f} kW",
                f"{MODEL2_R2:.4f}",
                f"{perf['model2_error'].mean():.4f} kW",
                f"{perf['model2_error'].min():.4f} kW",
                f"{perf['model2_error'].max():.4f} kW",
            ],
        }
    )

    st.dataframe(
        metrics_table,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Where does the remaining error appear?")

    regimes = pd.DataFrame(
        {
            "Operating regime": [
                "Night / low irradiance",
                "Daylight",
                "POA 0–100 W/m²",
                "POA 100–400 W/m²",
                "POA 400–800 W/m²",
                "POA 800–1200 W/m²",
            ],
            "MAE (kW)": [
                0.8703,
                25.7160,
                2.6303,
                28.0282,
                31.7641,
                23.9012,
            ],
        }
    )

    fig = px.bar(
        regimes,
        x="Operating regime",
        y="MAE (kW)",
        title="Error Concentration by Operating Regime",
    )

    st.plotly_chart(fig, width="stretch")

    st.caption(
        "The remaining error is not uniformly distributed. The difficult regime is active generation, "
        "especially moderate-to-high irradiance periods."
    )

    # ========================================================
    # TOP-GRADE ENHANCEMENT 1 — ACTUAL VS PREDICTED
    # ========================================================

    st.divider()
    st.subheader("Actual vs Predicted — Model 2")

    scatter_df = perf[
        ["timestamp", "actual", "model2_pred"]
    ].copy()

    scatter_df["actual"] = pd.to_numeric(
        scatter_df["actual"],
        errors="coerce",
    )

    scatter_df["model2_pred"] = pd.to_numeric(
        scatter_df["model2_pred"],
        errors="coerce",
    )

    scatter_df = scatter_df.dropna(
        subset=["actual", "model2_pred"]
    )

    if not scatter_df.empty:
        plot_min = min(
            scatter_df["actual"].min(),
            scatter_df["model2_pred"].min(),
        )

        plot_max = max(
            scatter_df["actual"].max(),
            scatter_df["model2_pred"].max(),
        )

        actual_pred_fig = go.Figure()

        actual_pred_fig.add_trace(
            go.Scatter(
                x=scatter_df["actual"],
                y=scatter_df["model2_pred"],
                mode="markers",
                name="Forecast records",
                marker=dict(
                    size=5,
                    opacity=0.45,
                ),
                customdata=scatter_df["timestamp"],
                hovertemplate=(
                    "Timestamp: %{customdata}<br>"
                    "Actual: %{x:.2f} kW<br>"
                    "Predicted: %{y:.2f} kW"
                    "<extra></extra>"
                ),
            )
        )

        actual_pred_fig.add_trace(
            go.Scatter(
                x=[plot_min, plot_max],
                y=[plot_min, plot_max],
                mode="lines",
                name="Perfect prediction",
                line=dict(
                    dash="dash",
                    width=2,
                ),
                hoverinfo="skip",
            )
        )

        actual_pred_fig.update_layout(
            title="Actual vs Predicted AC Power",
            xaxis_title="Actual AC Power (kW)",
            yaxis_title="Predicted AC Power (kW)",
            height=560,
        )

        st.plotly_chart(
            actual_pred_fig,
            width="stretch",
        )

        st.caption(
            "Points closer to the diagonal represent smaller prediction errors. "
            "The plot uses the saved Model 2 evaluation predictions."
        )

    # ========================================================
    # TOP-GRADE ENHANCEMENT 2 — ERROR DISTRIBUTION
    # ========================================================

    st.subheader("Prediction Error Distribution")

    error_values = pd.to_numeric(
        perf["model2_error"],
        errors="coerce",
    ).dropna()

    absolute_error_values = error_values.abs()

    if not error_values.empty:
        e1, e2, e3, e4 = st.columns(4)

        e1.metric(
            "Mean Error",
            f"{error_values.mean():.2f} kW",
        )

        e2.metric(
            "Median Absolute Error",
            f"{absolute_error_values.median():.2f} kW",
        )

        e3.metric(
            "90th Percentile |Error|",
            f"{absolute_error_values.quantile(0.90):.2f} kW",
        )

        e4.metric(
            "95th Percentile |Error|",
            f"{absolute_error_values.quantile(0.95):.2f} kW",
        )

        error_fig = px.histogram(
            error_values,
            nbins=60,
            title="Distribution of Prediction Error",
            labels={
                "value": "Prediction Error (Predicted − Actual) [kW]",
                "count": "Records",
            },
        )

        error_fig.add_vline(
            x=0,
            line_dash="dash",
        )

        error_fig.add_vline(
            x=float(error_values.mean()),
            line_dash="dot",
            annotation_text=(
                f"Mean = {error_values.mean():.2f} kW"
            ),
            annotation_position="top right",
        )

        error_fig.update_layout(
            xaxis_title="Prediction Error (kW)",
            yaxis_title="Number of Records",
            height=500,
        )

        st.plotly_chart(
            error_fig,
            width="stretch",
        )

        st.caption(
            "The distribution shows how prediction errors are spread around zero. "
            "A positive value means overprediction; a negative value means underprediction."
        )


# ============================================================
# ============================================================
# ============================================================
# 7. FORECAST DEMONSTRATION
# ============================================================

with tab_demo:
    st.header("Forecast Demonstration")
    st.caption(
        "Choose a historical timestamp T, inspect the exact dataset inputs, "
        "and compare the T+1 forecast with the actual generation."
    )

    st.info(
        "The historical replay uses the project's saved Model 2 evaluation prediction. "
        "This keeps the official result exactly aligned with the original evaluation."
    )

    # ========================================================
    # A. OFFICIAL DATASET REPLAY
    # ========================================================

    st.subheader("A. Official Dataset Replay")

    replay_perf = perf.copy()

    replay_min = replay_perf["timestamp"].min().date()
    replay_max = replay_perf["timestamp"].max().date()

    c1, c2 = st.columns(2)

    with c1:
        replay_date = st.date_input(
            "Forecast timestamp date",
            value=replay_min,
            min_value=replay_min,
            max_value=replay_max,
            key="replay_date_v5",
        )

    replay_hours = (
        replay_perf[
            replay_perf["timestamp"].dt.date == replay_date
        ]["timestamp"]
        .sort_values()
        .dt.strftime("%H:%M")
        .tolist()
    )

    with c2:
        if replay_hours:
            replay_time = st.selectbox(
                "Forecast hour T",
                replay_hours,
                key="replay_time_v5",
            )
        else:
            replay_time = None
            st.warning("No saved evaluation prediction is available for this date.")

    if replay_time is not None:
        replay_timestamp = pd.Timestamp(
            f"{replay_date} {replay_time}"
        )

        saved_row = replay_perf[
            replay_perf["timestamp"] == replay_timestamp
        ]

        input_row = model2_df[
            model2_df["timestamp"] == replay_timestamp
        ]

        if saved_row.empty or input_row.empty:
            st.error(
                "The selected timestamp is not available in both the saved "
                "prediction file and the Model 2 dataset."
            )
        else:
            saved_row = saved_row.iloc[0]
            input_row = input_row.iloc[0]

            official_forecast = float(saved_row["model2_pred"])
            actual = float(saved_row["actual"])
            signed_error = official_forecast - actual
            absolute_error = abs(signed_error)

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Current AC Power (T)",
                f"{float(input_row['ac_power__5069']):.2f} kW",
            )
            k2.metric(
                "Model 2 Forecast (T+1)",
                f"{official_forecast:.2f} kW",
            )
            k3.metric(
                "Actual AC Power (T+1)",
                f"{actual:.2f} kW",
            )
            k4.metric(
                "Absolute Error",
                f"{absolute_error:.2f} kW",
            )

            st.caption(
                f"{replay_timestamp} → {saved_row['target_timestamp']}"
            )

            if abs(actual) > 10:
                rel_error = absolute_error / abs(actual) * 100
                st.caption(
                    f"Single-record relative absolute error: {rel_error:.2f}%. "
                    "This is not the overall model accuracy."
                )
            else:
                st.caption(
                    "Percentage error is not reported because actual power is near zero."
                )

            st.subheader("Exact 27 inputs at T")

            feature_values = []
            for col in MODEL2_FEATURES:
                value = pd.to_numeric(
                    input_row[col],
                    errors="coerce",
                )
                feature_values.append(
                    value if pd.notna(value) else np.nan
                )

            feature_view = pd.DataFrame(
                {
                    "Feature": MODEL2_FEATURES,
                    "Value at T": feature_values,
                }
            )

            st.dataframe(
                feature_view,
                width="stretch",
                hide_index=True,
            )

            st.subheader("T → T+1")

            timeline = pd.DataFrame(
                {
                    "Stage": ["T input", "T+1 target"],
                    "Timestamp": [
                        str(replay_timestamp),
                        str(saved_row["target_timestamp"]),
                    ],
                    "AC Power": [
                        float(input_row["ac_power__5069"]),
                        actual,
                    ],
                    "Model 2 Forecast": [
                        np.nan,
                        official_forecast,
                    ],
                }
            )

            st.dataframe(
                timeline,
                width="stretch",
                hide_index=True,
            )

            st.success(
                "Forecast shown above is the saved Model 2 evaluation result from the project."
            )

    st.divider()

    # ========================================================
    # B. MANUAL / WHAT-IF PREDICTION
    # ========================================================

    st.subheader("B. Manual Scenario Prediction")

    st.write(
        "Here you can start from a real dataset row and optionally change the inputs. "
        "When the values are unchanged, the dashboard reports the saved dataset prediction. "
        "After edits, the cached Model 2 is used to generate a scenario estimate."
    )

    st.warning(
        "**Evaluation boundary:** Model 2 was tested on the **2017–2018 confirmatory period "
        "(5,522 records)**. This manual demonstration therefore uses input records from that "
        "same evaluated period so the scenario remains aligned with the project's reported test performance. "
        "Timestamps outside the evaluated Model 2 period are not forecast here."
    )

    manual_eval_min = perf["timestamp"].min()
    manual_eval_max = perf["timestamp"].max()

    manual_timestamp = st.datetime_input(
        "Timestamp T",
        value=pd.Timestamp("2017-05-27 10:00").to_pydatetime(),
        min_value=manual_eval_min.to_pydatetime(),
        max_value=manual_eval_max.to_pydatetime(),
        key="manual_timestamp_v5",
    )

    timestamp_key = pd.Timestamp(manual_timestamp).strftime("%Y%m%d_%H%M")
    selected_timestamp = pd.Timestamp(manual_timestamp)

    # IMPORTANT: Use only an exact Model 2 forecasting timestamp from the evaluated period.
    # Never substitute the nearest available row because that would combine
    # inputs from a different timestamp with the user's selected timestamp.
    selected_rows = model2_df.loc[
        model2_df["timestamp"] == selected_timestamp
    ]

    evaluated_rows = perf.loc[
        perf["timestamp"] == selected_timestamp
    ]

    if selected_rows.empty or evaluated_rows.empty:
        st.warning(
            f"No evaluated Model 2 forecast is available for "
            f"{selected_timestamp:%Y-%m-%d %H:%M}."
        )
        st.caption(
            "The selected timestamp is not an exact record in the evaluated 2017–2018 "
            "Model 2 period. The dashboard does not substitute a nearby timestamp, so "
            "no forecasting is performed for this selection. A valid current-hour AC input "
            "and a valid next-hour target pair are required."
        )

    else:
        defaults = selected_rows.iloc[0]
        selected_model2_timestamp = pd.Timestamp(defaults["timestamp"])

        st.caption(
            f"Inputs are initialized from the exact Model 2 dataset record: "
            f"{selected_model2_timestamp:%Y-%m-%d %H:%M}."
        )
        st.caption(
            "All scenario inputs shown here are therefore drawn from the evaluated 2017–2018 "
            "Model 2 period; edited values remain what-if inputs rather than new test results."
        )

        cols = st.columns(3)

        with cols[0]:
            manual_ac = st.number_input(
                "Current AC power (kW)",
                value=float(defaults["ac_power__5069"]),
                step=1.0,
                key=f"manual_ac_{timestamp_key}",
            )
            manual_ambient = st.number_input(
                "Ambient temperature",
                value=float(defaults["ambient_temp__5062"]),
                step=0.1,
                key=f"manual_ambient_{timestamp_key}",
            )
            manual_module = st.number_input(
                "Module temperature",
                value=float(defaults["module_temp__5063"]),
                step=0.1,
                key=f"manual_module_{timestamp_key}",
            )
            manual_poa = st.number_input(
                "POA irradiance (W/m²)",
                value=float(defaults["poa_irradiance__5061"]),
                step=1.0,
                key=f"manual_poa_{timestamp_key}",
            )
            manual_t2m = st.number_input(
                "T2M",
                value=float(defaults["T2M"]),
                step=0.1,
                key=f"manual_t2m_{timestamp_key}",
            )
            manual_rh = st.number_input(
                "RH2M",
                value=float(defaults["RH2M"]),
                step=0.1,
                key=f"manual_rh_{timestamp_key}",
            )
            manual_ws = st.number_input(
                "WS10M",
                value=float(defaults["WS10M"]),
                step=0.1,
                key=f"manual_ws_{timestamp_key}",
            )
            manual_wd = st.number_input(
                "WD10M",
                value=float(defaults["WD10M"]),
                step=1.0,
                key=f"manual_wd_{timestamp_key}",
            )
            manual_ps = st.number_input(
                "PS",
                value=float(defaults["PS"]),
                step=0.1,
                key=f"manual_ps_{timestamp_key}",
            )

        with cols[1]:
            manual_precip = st.number_input(
                "PRECTOTCORR",
                value=float(defaults["PRECTOTCORR"]),
                step=0.1,
                key=f"manual_precip_{timestamp_key}",
            )
            manual_allsky = st.number_input(
                "ALLSKY_SFC_SW_DWN",
                value=float(defaults["ALLSKY_SFC_SW_DWN"]),
                step=0.1,
                key=f"manual_allsky_{timestamp_key}",
            )
            manual_cloud = st.number_input(
                "CLOUD_AMT",
                value=float(defaults["CLOUD_AMT"]),
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key=f"manual_cloud_{timestamp_key}",
            )
            manual_high = st.number_input(
                "High cloud cover",
                value=float(defaults["high_cloud_cover"]),
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key=f"manual_high_{timestamp_key}",
            )
            manual_medium = st.number_input(
                "Medium cloud cover",
                value=float(defaults["medium_cloud_cover"]),
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key=f"manual_medium_{timestamp_key}",
            )
            manual_low = st.number_input(
                "Low cloud cover",
                value=float(defaults["low_cloud_cover"]),
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key=f"manual_low_{timestamp_key}",
            )
            manual_gust = st.number_input(
                "Wind gust 10m",
                value=float(defaults["wind_gust_10m"]),
                step=0.1,
                key=f"manual_gust_{timestamp_key}",
            )
            manual_snow = st.number_input(
                "Snowfall",
                value=float(defaults["snowfall"]),
                step=0.1,
                key=f"manual_snow_{timestamp_key}",
            )
            manual_wind900 = st.number_input(
                "900 hPa wind speed",
                value=float(defaults["wind_speed_900_mb"]),
                step=0.1,
                key=f"manual_wind900_{timestamp_key}",
            )

        with cols[2]:
            manual_dir900 = st.number_input(
                "900 hPa wind direction",
                value=float(defaults["wind_direction_900_mb"]),
                step=1.0,
                key=f"manual_dir900_{timestamp_key}",
            )

            st.info(
                "Solar position and cyclic time features are calculated automatically "
                "from the selected timestamp."
            )

            if st.button(
                "Predict Next Hour",
                type="primary",
                key=f"manual_predict_{timestamp_key}",
            ):
                naive = pd.Timestamp(manual_timestamp)
                mst_timestamp = naive.tz_localize("Etc/GMT+7")

                solar = pvlib.solarposition.get_solarposition(
                    time=mst_timestamp,
                    latitude=39.7404,
                    longitude=-105.1719,
                )

                zenith = float(solar["zenith"].iloc[0])
                azimuth = float(solar["azimuth"].iloc[0])

                hour_float = naive.hour + naive.minute / 60.0
                day_of_year = naive.dayofyear

                manual_values = {
                    "ac_power__5069": manual_ac,
                    "ambient_temp__5062": manual_ambient,
                    "module_temp__5063": manual_module,
                    "poa_irradiance__5061": manual_poa,
                    "T2M": manual_t2m,
                    "RH2M": manual_rh,
                    "WS10M": manual_ws,
                    "WD10M": manual_wd,
                    "PS": manual_ps,
                    "PRECTOTCORR": manual_precip,
                    "ALLSKY_SFC_SW_DWN": manual_allsky,
                    "CLOUD_AMT": manual_cloud,
                    "high_cloud_cover": manual_high,
                    "medium_cloud_cover": manual_medium,
                    "low_cloud_cover": manual_low,
                    "wind_gust_10m": manual_gust,
                    "snowfall": manual_snow,
                    "wind_speed_900_mb": manual_wind900,
                    "wind_direction_900_mb": manual_dir900,
                    "solar_zenith_angle": zenith,
                    "solar_azimuth_angle": azimuth,
                    "hour_sin": np.sin(2 * np.pi * hour_float / 24),
                    "hour_cos": np.cos(2 * np.pi * hour_float / 24),
                    "month_sin": np.sin(2 * np.pi * naive.month / 12),
                    "month_cos": np.cos(2 * np.pi * naive.month / 12),
                    "day_of_year_sin": np.sin(
                        2 * np.pi * day_of_year / 365.25
                    ),
                    "day_of_year_cos": np.cos(
                        2 * np.pi * day_of_year / 365.25
                    ),
                }

                manual_X = pd.DataFrame(
                    [manual_values],
                    columns=MODEL2_FEATURES,
                ).astype(float)

                # Check whether the manual values are exactly the selected
                # Model 2 dataset row. In that case, use the saved official prediction.
                dataset_vector = (
                    defaults[MODEL2_FEATURES]
                    .apply(pd.to_numeric, errors="coerce")
                    .astype(float)
                )

                manual_vector = manual_X.iloc[0]

                same_as_dataset = np.allclose(
                    manual_vector.values,
                    dataset_vector.values,
                    rtol=1e-9,
                    atol=1e-9,
                    equal_nan=True,
                )

                if same_as_dataset:
                    saved_match = perf[
                        perf["timestamp"] == selected_model2_timestamp
                    ]

                    if not saved_match.empty:
                        scenario_prediction = float(
                            saved_match.iloc[0]["model2_pred"]
                        )
                        scenario_actual = float(
                            saved_match.iloc[0]["actual"]
                        )
                        scenario_error = abs(
                            scenario_prediction - scenario_actual
                        )

                        st.success(
                            f"Dataset-matched prediction: **{scenario_prediction:.2f} kW**"
                        )
                        st.caption(
                            f"This matches the saved Model 2 evaluation record at "
                            f"{selected_model2_timestamp:%Y-%m-%d %H:%M}."
                        )

                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            "Saved Forecast (T+1)",
                            f"{scenario_prediction:.2f} kW",
                        )
                        m2.metric(
                            "Actual (T+1)",
                            f"{scenario_actual:.2f} kW",
                        )
                        m3.metric(
                            "Absolute Error",
                            f"{scenario_error:.2f} kW",
                        )
                    else:
                        st.warning(
                            "The inputs match a Model 2 dataset row, but no saved prediction "
                            "record exists for this timestamp."
                        )
                else:
                    model = train_model2()

                    scenario_prediction = float(
                        model.predict(
                            manual_X,
                            num_iteration=model.best_iteration_,
                        )[0]
                    )

                    st.success(
                        f"Scenario forecast for T+1: **{scenario_prediction:.2f} kW**"
                    )

                    st.warning(
                        "Because at least one input was manually changed, this is a "
                        "what-if model estimate. It is not an official saved evaluation prediction."
                    )

# 8. ERROR ANALYSIS
# ============================================================

with tab_errors:
    st.header("Error Analysis")
    st.caption(
        "The purpose is not to hide the 7–8% unexplained variation, but to show where it is concentrated."
    )

    regimes = pd.DataFrame(
        {
            "Regime": [
                "Night / low irradiance",
                "Daylight",
                "POA 0–100",
                "POA 100–400",
                "POA 400–800",
                "POA 800–1200",
            ],
            "MAE (kW)": [
                0.8703,
                25.7160,
                2.6303,
                28.0282,
                31.7641,
                23.9012,
            ],
            "Mean bias (kW)": [
                0.0740,
                9.8300,
                0.2074,
                13.4203,
                13.8408,
                -8.8180,
            ],
        }
    )

    fig = px.bar(
        regimes,
        x="Regime",
        y="MAE (kW)",
        hover_data=["Mean bias (kW)"],
        title="Where prediction error concentrates",
    )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Largest observed errors in the confirmatory period")

    top20 = (
        perf[
            [
                "timestamp",
                "target_timestamp",
                "actual",
                "model2_pred",
                "model2_error",
                "model2_abs_error",
                "poa_irradiance__5061",
                "ambient_temp__5062",
                "module_temp__5063",
                "CLOUD_AMT",
                "solar_zenith_angle",
            ]
        ]
        .sort_values("model2_abs_error", ascending=False)
        .head(20)
    )

    st.dataframe(
        top20,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "A large residual describes a prediction miss; it does not by itself prove a plant fault, curtailment event or sensor failure."
    )


# ============================================================
# 9. EXPLAINABILITY
# ============================================================

with tab_explain:
    st.header("Explainability")

    shap_top = (
        shap_global[
            ["feature", "mean_abs_shap", "importance_percentage"]
        ]
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
    )

    gain_top = (
        gain_global[
            ["feature", "gain_percentage"]
        ]
        .sort_values("gain_percentage", ascending=False)
        .head(15)
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            shap_top.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Global SHAP Importance",
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = px.bar(
            gain_top.sort_values("gain_percentage"),
            x="gain_percentage",
            y="feature",
            orientation="h",
            title="LightGBM Gain Importance",
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("How should these two explanations be read?")

    explain_compare = pd.DataFrame(
        {
            "Method": [
                "SHAP",
                "LightGBM gain",
                "Both together",
            ],
            "What it tells us": [
                "How much each feature contributes to individual predictions on average across the evaluated data.",
                "How much a feature contributes to reducing tree-model loss across the fitted ensemble.",
                "A stronger cross-check of model reliance than treating either importance ranking as a causal statement.",
            ],
        }
    )

    st.dataframe(
        explain_compare,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Additional interpretation")

    st.write(
        "A feature can be important without being a causal driver of generation. "
        "Solar irradiance, temperature and operating-state variables can also be correlated with one another, "
        "so importance should be interpreted as the model's predictive use of the available signals, not as a "
        "physical attribution of cause and effect."
    )

    st.write(
        "The global charts describe the model across the evaluated population. They do not explain why one "
        "specific timestamp received a particular forecast. A future local-SHAP or prediction-decomposition layer "
        "could provide timestamp-level explanations such as which inputs pushed an individual forecast upward or downward."
    )

    st.info(
        "Feature importance explains model behaviour; it is not a causal proof, a fault diagnosis, "
        "or evidence that changing one feature alone would produce the same change in plant output."
    )


# ============================================================
# 10. METHODOLOGY & SCOPE
# ============================================================

with tab_method:
    st.header("Methodology & Honest Scope")

    architecture = pd.DataFrame(
        {
            "Phase": [
                "Data acquisition",
                "Data integration",
                "Data audit",
                "Preprocessing",
                "Forecast dataset",
                "EDA",
                "Model 1 research",
                "Model 2",
                "Explainability",
                "Performance analytics",
                "presenter dashboard",
            ],
            "What was done": [
                "Real NREL PVDAQ System 1433 selected",
                "NASA POWER + ERA5 + 900 hPa wind integrated",
                "Missingness, duplicates, range and physical anomalies inspected",
                "Evidence-based cleaning and ML exclusions",
                "Strict 1-hour timestamp continuity for T→T+1 pairs",
                "Descriptive + temporal + physical + regime analysis",
                "20 variants explored to improve weather/solar generalization",
                "Current AC added for intraday T+1 forecasting",
                "SHAP + LightGBM gain",
                "Hour, month, irradiance, cloud, residual analysis",
                "Presentation-first story + historical forecast demonstration",
            ],
        }
    )

    st.dataframe(
        architecture,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("What this project proves")

    st.success(
        "A real-data, plant-specific next-hour forecasting workflow can be built, "
        "validated, explained and demonstrated end-to-end."
    )

    st.subheader("What this project does not yet prove")

    for text in [
        "It does not prove universal cross-plant generalization without retraining/calibration.",
        "It is not a live SCADA or real-time dispatch system.",
        "Historical NASA/ERA5 observations are not operational future weather forecasts.",
        "R² = 0.9235 is not a per-record 92.35% accuracy score.",
        "2017–2018 is best described as a confirmatory evaluation period, not a pristine untouched test set.",
    ]:
        st.write(f"• {text}")

    st.divider()

    st.subheader("Forecast-horizon evidence from autocorrelation")

    autocorr_horizon = pd.DataFrame(
        {
            "Lag": [1, 2, 3, 6, 12, 24],
            "AC correlation": [0.915, 0.761, 0.569, -0.023, -0.411, 0.838],
            "Interpretation": [
                "Very strong short-term state persistence",
                "Strong short-term persistence remains",
                "Useful short-term persistence remains",
                "Near-zero same-state linear persistence",
                "Negative relationship at this lag",
                "Strong daily periodic structure returns",
            ],
        }
    )

    h1, h2 = st.columns(2)

    with h1:
        fig = px.line(
            autocorr_horizon,
            x="Lag",
            y="AC correlation",
            markers=True,
            title="Observed AC Power Autocorrelation",
            labels={"Lag": "Lag (hours)", "AC correlation": "Correlation"},
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, width="stretch")

    with h2:
        st.dataframe(
            autocorr_horizon,
            width="stretch",
            hide_index=True,
        )

    st.write(
        "The observed autocorrelation shows why the current Model 2 design is most naturally a short-horizon, "
        "state-aware model. Correlation is very strong at 1–3 hours, while the same-hour linear persistence is "
        "already near zero by the 6-hour lag. This does **not** create a mathematical 4-hour cutoff, but it does "
        "show that the current plant-state signal should not be assumed to remain equally informative for long horizons."
    )

    st.caption(
        "The current project validates Model 2 at T+1 only. Longer-horizon performance must be trained and evaluated "
        "explicitly rather than inferred from the T+1 result."
    )

    st.divider()

    st.subheader("Potential next-generation extension: multi-horizon hybrid forecasting")

    st.write(
        "A stronger operational architecture would not force one model to solve every horizon. Instead, it could "
        "combine the strengths of the current Model 2 and the Model 1 weather/solar branch, with different information "
        "sources becoming more important as the horizon increases."
    )

    horizon_plan = pd.DataFrame(
        {
            "Forecast horizon": [
                "T+1 to approximately T+4",
                "Beyond T+4 to intraday / day-ahead",
                "Fusion layer across horizons",
            ],
            "Proposed approach": [
                "State-aware Model 2 branch using recent SCADA/plant state + current weather/solar variables.",
                "Model 1-style weather/solar branch driven by actual future weather forecasts, cloud forecasts/nowcasts and solar geometry.",
                "Horizon-aware fusion using plant health, availability, forecast uncertainty and data-quality signals.",
            ],
            "Why": [
                "Short-horizon generation retains strong dependence on the current operating state.",
                "Current AC power becomes progressively less informative as the horizon extends, so future exogenous information becomes more important.",
                "A hybrid design can shift the balance from live plant state toward forecast weather as the horizon increases.",
            ],
        }
    )

    st.dataframe(
        horizon_plan,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
        **Illustrative architecture**

        **Live time T** → **SCADA + weather + plant-health state** → **short-horizon branch** → **T+1…T+4**

        **Future weather forecasts + satellite/cloud nowcasts + solar geometry** → **weather/solar branch** → **T+4…day-ahead**

        **Horizon + health + availability + uncertainty** → **fusion / calibration layer** → **final multi-horizon forecast**
        """
    )

    st.subheader("Real-world signals that could strengthen the next generation")

    factors = pd.DataFrame(
        {
            "Signal group": [
                "Live SCADA operating state",
                "Plant health / health coefficient",
                "Inverter and electrical state",
                "PV performance / degradation",
                "Weather and atmosphere",
                "Cloud movement / nowcasting",
                "Plant availability / operational events",
                "Data quality",
            ],
            "Examples": [
                "Live AC/DC power, voltage, current, irradiance, module temperature, inverter status and recent ramps.",
                "A health index derived from efficiency, availability, repeated alarms, thermal behaviour and recent performance relative to expected output.",
                "Inverter operating mode, clipping/limiting, DC-side voltage/current, string or combiner behaviour and reactive/grid constraints where available.",
                "Degradation trend, soiling, cleaning history, tracker position, snow cover and maintenance effects where measurable.",
                "Forecast temperature, humidity, wind, pressure, precipitation, irradiance and cloud variables from operational forecast products.",
                "Satellite imagery, cloud-motion features and short-term cloud-cover/irradiance nowcasts for rapidly changing conditions.",
                "Curtailment, outages, maintenance windows, grid availability, inverter trips and plant-level availability signals.",
                "Sensor health, missingness, stale readings, quality flags and confidence indicators so the model knows when inputs are unreliable.",
            ],
        }
    )

    st.dataframe(
        factors,
        width="stretch",
        hide_index=True,
    )

    st.subheader("How the next-generation design could address the Model 1 weakness")

    st.write(
        "The Model 1 result shows that adding more engineered features alone did not guarantee transfer to the later "
        "period. A next-generation version should therefore change the information available to the model, not only "
        "the algorithm. In particular, the weather/solar branch should use genuine future forecast inputs, while a "
        "plant-health and availability state should represent conditions that historical reanalysis cannot describe."
    )

    improvement_points = [
        "Replace historical weather observations used as retrospective inputs with forecast weather variables for each future horizon.",
        "Add live SCADA state and a calibrated plant-health / availability coefficient so the model can distinguish weather-driven loss from equipment or operating-state loss.",
        "Use horizon-specific direct models or a carefully validated hybrid strategy instead of assuming one T+1 model can be rolled forward unchanged.",
        "Add uncertainty intervals so the system reports both a forecast and the confidence associated with changing weather or plant conditions.",
        "Validate with strict walk-forward evaluation and, eventually, on additional plants or operating periods to test transferability.",
    ]

    for item in improvement_points:
        st.write(f"• {item}")

    st.info(
        "This is a proposed next-generation architecture, not a result already achieved in the current project. "
        "Its performance would need new data pipelines, future-weather inputs, SCADA integration and fresh walk-forward validation."
    )


# FOOTER
# ============================================================

st.divider()

st.caption(
    "Solar Power Forecasting & Plant Analytics • NREL PVDAQ System 1433 • "
    "Historical next-hour forecasting • Project presentation dashboard"
)


