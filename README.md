# Solar Power Generation Forecasting & Plant Performance Analytics System

An end-to-end data science and machine learning project for **NREL PVDAQ System 1433 (RSF1, Golden, Colorado)**.

The project combines real photovoltaic plant measurements with **NASA POWER** and **ERA5** environmental data to build a transparent workflow from data acquisition and quality auditing to exploratory analysis, next-hour forecasting, model research, explainability, performance analytics, and an interactive Streamlit dashboard.

## 🚀 Live Dashboard

**[Open the Live Streamlit Dashboard](https://solar-forecasting-j6dxtgxbemmbvyldc3jbl9.streamlit.app/)**

The dashboard presents the complete project journey, including:

- Data acquisition and integration
- Data-quality and preprocessing decisions
- Deep EDA insights
- Model 1 research experiments
- Final Model 2 results
- Actual vs predicted analysis
- Prediction error analysis
- SHAP and feature importance
- Historical forecast demonstration
- Methodology, limitations, and future multi-horizon architecture

---

## 📌 Project Objective

The main forecasting question is:

> **Given the information available at hour T, can we estimate the plant's AC power at T+1?**

The project investigates two forecasting approaches:

### Model 1 — Weather / Solar Forecasting

Uses **26 weather, irradiance, atmospheric, solar-position, and temporal features** without current AC power.

**Goal:** evaluate how well a weather/solar-driven model can generalize without directly observing the current plant operating state.

### Model 2 — Intraday Forecasting

Uses the same 26 features **plus current-hour AC power**, giving a total of **27 input features**.

**Goal:** estimate next-hour plant AC power using the current operating state together with environmental conditions.

Model 2 is therefore an **intraday T+1 forecasting model**, not a day-ahead model.

---

## 🗂️ Data Sources

### NREL PVDAQ

Real PV plant measurements from:

**PVDAQ System 1433 — RSF1, Golden, Colorado**

The original PVDAQ data were recorded at 15-minute resolution and were aggregated to hourly observations for the modelling pipeline.

### NASA POWER

Historical environmental variables including:

- Air temperature
- Relative humidity
- Wind speed and direction
- Surface pressure
- Precipitation
- Surface solar radiation

### ERA5

Additional atmospheric and cloud information including:

- High, medium, and low cloud cover
- Wind gust
- Snowfall
- Additional atmospheric wind information

### Important note

NASA POWER and ERA5 variables used in this project are **historical observations/reanalysis inputs**. They are not operational future weather forecasts.

Therefore, the current Model 1 should **not** be described as a true operational day-ahead forecast.

---

## 🔄 End-to-End Data Pipeline

```text
Real PVDAQ Plant Data
        │
        ▼
15-Minute Measurements
        │
        ▼
Hourly Aggregation
        │
        ▼
NASA POWER Integration
        │
        ▼
ERA5 Integration
        │
        ▼
900 hPa Wind Features
        │
        ▼
Solar Position Features
        │
        ▼
Data Audit
        │
        ▼
Evidence-Based Preprocessing
        │
        ▼
ML-Ready Dataset
        │
        ├──────────────► Model 1 T+1
        │
        └──────────────► Model 2 T+1
                              │
                              ▼
                    Evaluation + Explainability
                              │
                              ▼
                       Streamlit Dashboard



📊 Dataset Transformation

The project moves through four main dataset stages:
1. Integrated Master Dataset

**File:** `PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv`  
**Shape:** `61,726 × 25`
Final integrated hourly dataset combining:
- Real PVDAQ plant measurements
- NASA POWER environmental variables
- ERA5 cloud and atmospheric variables
- 900 hPa wind features
**Purpose:** Serves as the main integrated plant + environmental master dataset.

2. ML-Ready Dataset

**File:** `PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv`  
**Shape:** `56,980 × 28`

Produced after evidence-based:
- Missing-value handling
- Physical-value checks
- Sensor-quality decisions
- Solar-position feature generation
- ML feature preparation

**Purpose:** Provides the cleaned and traceable feature space used to construct the forecasting datasets.

3. Model 1 Forecasting Dataset

**File:** `Model1_WeatherSolar_NextHour.csv`  
**Shape:** `56,275 × 29`

**Input:** 26 weather, irradiance, solar-position, and temporal features  
**Target:** Next-hour AC power (`T+1`)

**Purpose:** Evaluate weather/solar-only next-hour forecasting without current AC power.

4. Model 2 Forecasting Dataset

**File:** `Model2_Intraday_NextHour.csv`  
**Shape:** `56,275 × 30`

**Input:** 27 features = 26 Model 1 features + current-hour AC power  
**Target:** Next-hour AC power (`T+1`)

**Purpose:** Intraday next-hour forecasting using both environmental conditions and the current plant operating state.

Deployment Note:

The public deployment repository contains only the processed datasets required by the dashboard. Raw plant and reanalysis files are intentionally excluded from the deployed application.



🧹 Data Quality & Preprocessing
The project did not apply blind interpolation or blanket outlier removal.
Major evidence-based decisions included:
- Missing AC power was not replaced with zero.
- Long missing target periods were excluded from supervised forecasting.
- Rows with missing core plant sensors were excluded where required.
- DC voltage was excluded from the primary ML feature set because of high missingness.
- Raw PR was excluded because of substantial missingness and implausible extreme values.
- Negative AC and negative POA observations were investigated contextually before final ML treatment.
- Duplicate timestamps were checked and none were present.
- The original integrated master dataset was preserved for traceability.
The preprocessing stage therefore separates:
original evidence → audit → decision → ML-ready copy
rather than silently changing the raw data.


🔎 Exploratory Data Analysis
Deep EDA was used to understand:
- Hourly and seasonal generation behaviour
- AC power autocorrelation
- Generation ramps
- POA irradiance response
- Cloud-cover effects
- Temperature relationships
- Daylight vs night behaviour
- Missingness patterns
- Sensor relationships
- Energy consistency
- Operating regimes
- Capacity-utilization behaviour
Important temporal observations include:
- 1-hour AC autocorrelation ≈ 0.915
- 2-hour autocorrelation ≈ 0.761
- 3-hour autocorrelation ≈ 0.569
- 6-hour autocorrelation ≈ -0.023
- 24-hour autocorrelation ≈ 0.838
These observations influenced the forecasting design and the proposed future multi-horizon architecture.


🤖 Model 1 Research
Model 1 was treated as a research problem rather than a single model.
A total of 20 variants (V1–V20) were investigated, including experiments involving:
- Time and solar features
- Temporal lags
- Regularization
- Operating-regime modelling
- Recent-history training
- XGBoost
- Temporal weather features
- Clear-sky features
- Ensembles
- Physics-informed features
- Two-stage classification/regression
- LSTM sequence modelling
- Distribution-shift diagnosis
- Recency weighting
- Target transformations
- Residual calibration

Main research finding
Several Model 1 variants improved rolling validation performance, but those gains did not transfer reliably to the later confirmatory period.
This validation-to-confirmatory gap became an important project finding rather than something hidden from the final analysis.


⚡ Model 2 — Final Intraday Forecasting Model
Model 2 uses:
27 features → next-hour AC power
The algorithm used is LightGBM regression.
Current AC power is included because the model represents an intraday operating-state forecasting problem.
Evaluation Scope
Evaluation period: 2017–2018 confirmatory period
Evaluation records: 5,522

The evaluation period should be described as a confirmatory period, not a pristine untouched test set, because the period was inspected during earlier Model 1 experimentation.
Final Model 2 Performance
Metric	Model 2
MAE	12.2133 kW
RMSE	24.3949 kW
R²	0.9235


Persistence baseline:
Metric	Persistence
MAE	17.6197 kW
RMSE	35.7158 kW
R²	0.8360


Model 2 reduced:
- MAE by approximately 30.68%
- RMSE by approximately 31.70%
compared with the persistence baseline.
Important interpretation
R² = 0.9235 does not mean 92.35% per-record accuracy.
R² is a variance-explained metric. MAE and RMSE provide the practical forecast-error interpretation.


📈 Error Analysis
The project does not rely on a single aggregate metric.
Error analysis includes:
- Actual vs predicted AC power
- Error distribution
- Error over time
- Hourly performance
- Monthly performance
- Irradiance-regime performance
- Cloud-regime performance
- Daylight vs night performance
- Top-error observations
- Comparison against persistence

A major observation is that remaining errors are concentrated more strongly during active generation, especially moderate-to-high irradiance conditions.


🧠 Explainability
Model 2 explainability uses:
- SHAP global importance
- LightGBM feature importance
- Feature-level model diagnostics

The dashboard distinguishes between:
feature importance → model behaviour
and
causal interpretation → not established by feature importance alone
Therefore, importance scores are used to understand how the model relies on available signals, not as proof that a feature physically causes a specific plant behaviour.


🖥️ Interactive Dashboard
The Streamlit dashboard brings the complete project together in one place.
Main sections
1. Executive Overview
2. Data Journey
3. EDA Insights
4. Forecast Design
5. Model 1 Research
6. Model 2 Results
7. Forecast Demonstration
8. Error Analysis
9. Explainability
10. Methodology & Scope

The forecast demonstration supports historical replay and controlled what-if scenarios while maintaining the evaluated Model 2 timestamp boundary.
🔮 Honest Scope & Future Multi-Horizon Architecture
The current project validates next-hour forecasting.
The autocorrelation analysis shows strong short-term temporal dependence, but it does not establish a mathematical hard four-hour forecasting limit.
The future system is therefore envisioned as a multi-horizon forecasting architecture rather than simply extending the same model repeatedly.

Short horizon — approximately T+1 to T+4
A future short-horizon model could combine:
- Current and recent SCADA measurements
- Current AC and DC operating state
- Inverter status
- Plant health indicators
- Rolling generation ramps
- Recent irradiance and temperature
- Cloud movement / nowcasting
- Real-time sensor quality information

Longer horizon — beyond short intraday horizons
Longer forecasts would rely more strongly on:
- Future weather forecasts
- Forecast irradiance
- Cloud-cover forecasts
- Solar geometry
- Temperature forecasts
- Wind forecasts
- Snow conditions
- Atmospheric variables

Proposed multi-horizon fusion
A future architecture could combine:
                 ┌─────────────────────────┐
                 │ Weather / Solar Model   │
                 │ Model 1 style inputs    │
                 └────────────┬────────────┘
                              │
                              ▼
                     Horizon-aware Fusion
                              ▲
                              │
                 ┌────────────┴────────────┐
                 │ Plant-State Model       │
                 │ Model 2 / SCADA inputs  │
                 └──────────────────────────┘

The relative contribution of each branch could depend on forecast horizon.
Real-world plant factors for the next generation
A production-oriented system should also consider:
- Live SCADA readings
- Inverter operating state
- Plant health coefficient
- Equipment degradation
- Soiling
- Tracker position
- Snow accumulation
- Curtailment
- Grid availability
- Maintenance periods
- Sensor quality
- Communication outages
- Cloud nowcasting
- Plant availability
- Operational alarms

These additions could help address a major weakness observed in Model 1: weather and solar variables alone may not capture the full current operating state of a PV plant.
Important boundary
This future architecture is a proposed next-generation extension, not a model already validated by this project.

📁 Repository Structure
Solar-Forecasting/
│
├── 46_interviewer_solar_forecasting_dashboard_v8.py
├── requirements.txt
├── README.md
│
├── data/
│   └── processed/
│       ├── Model1_WeatherSolar_NextHour.csv
│       ├── Model2_Intraday_NextHour.csv
│       └── PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv
│
├── reports/
│   ├── eda/
│   ├── explainability/
│   ├── performance_analytics/
│   └── master_dataset_audit.json
│
└── src/
    ├── preprocessing
    ├── EDA
    ├── Model 1 research
    ├── Model 2 analytics
    ├── NASA / ERA5 integration
    └── final evaluation


▶️ Run Locally
Clone the repository:
git clone https://github.com/ayush13122/Solar-Forecasting.git
cd Solar-Forecasting

Create and activate a Python environment, then install dependencies:
pip install -r requirements.txt

Run the dashboard:
streamlit run 46_interviewer_solar_forecasting_dashboard_v8.py


⚠️ Scientific Scope
This project is:
- Plant-specific to NREL PVDAQ System 1433
- Historical
- Focused on next-hour forecasting
- Not a live SCADA system
- Not a universal cross-plant predictor
- Not yet an operational day-ahead forecasting system
Historical NASA POWER and ERA5 observations/reanalysis should not be interpreted as future weather forecasts.
The project focuses on building a transparent, reproducible and scientifically defensible forecasting workflow rather than claiming production deployment readiness.


📌 Project Status

Current status: End-to-end project completed for historical next-hour forecasting and analytics.
Completed:
- Real-data acquisition
- Multi-source data integration
- Data audit
- Evidence-based preprocessing
- Deep EDA
- Model 1 research and diagnostics
- Model 2 LightGBM forecasting
- Error and performance analytics
- SHAP explainability
- Interactive Streamlit dashboard
- GitHub repository
- Streamlit Cloud deployment


Future work:
- Operational future weather forecasts
- Multi-horizon forecasting
- Live SCADA integration
- Plant-health-aware modelling
- Cross-plant validation
- Forecast uncertainty estimation
- Production monitoring and retraining
