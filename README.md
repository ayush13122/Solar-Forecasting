# Solar Power Generation Forecasting & Plant Performance Analytics System

This project studies **PVDAQ System 1433 (RSF1, Golden, Colorado)** and joins
its plant measurements with NASA POWER and ERA5 weather features. The goal is
to build an honest, explainable progression from data quality checks to EDA,
power-generation modelling, and plant underperformance analysis.

## Verified data layers

| Layer | Location | Shape | Purpose |
| --- | --- | ---: | --- |
| Raw PVDAQ | `data/raw/PVDAQ_System_1433.csv` | 246,903 rows x 10 columns | Original plant readings; never modify it. |
| Final master | `../PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv` | 61,726 rows x 25 columns | Hourly analysis and modelling source. |

The final master is stored one directory above this project folder because it
was produced during the earlier data-integration stage. Scripts resolve that
path automatically; do not move, rename, or overwrite the file.

## First reproducible step: data audit

From this folder, run:

```powershell
python src/audit_final_dataset.py
```

It creates, without modifying either CSV:

- `data/processed/final_dataset_quality_report.json` — structured results for later code.
- `reports/01_data_quality_audit.md` — a readable project record.

## Exploratory data analysis

After the audit, run:

```powershell
python src/exploratory_analysis.py
```

This produces an EDA report, reusable numeric metrics, and four figures in
`reports/figures/`. It uses only complete PVDAQ target hours for the core
relationships and clearly separates same-time diagnostics from future forecasts.

## Roadmap

1. Verify the final master dataset. ✅
2. Exploratory data analysis: generation patterns, irradiance and weather relationships, missing-data patterns. ✅
3. Create transparent features and a time-aware train/test split.
4. Compare Linear Regression, Random Forest, and a gradient-boosting model.
5. Use prediction residuals to flag potential plant underperformance.
6. Build the interactive Streamlit analytics dashboard.

## Important terminology

Predicting power using weather recorded at the **same timestamp** is
regression, not future forecasting. Future forecasting will require lagged
generation and weather inputs available before the prediction time. We will
label each model correctly in the final report and dashboard.
