# C2C – Clutter 2 Clarity

An end-to-end, **agentic** data-cleaning pipeline that transforms messy, real-world tables into analysis-ready datasets—no coding required.

---

## 🚀 Key Features

- **Sense**  
  - Statistical profiling: missing %, duplicates, skewness, kurtosis  
  - Isolation Forest anomaly detection  
- **Think**  
  - Lightweight adaptive controller tunes imputation (mean/median) and outlier strategy (cap/remove/skip) using run-by-run metrics  
- **Act**  
  - Skew-aware imputation (mean or median)  
  - Duplicate removal  
  - Categorical normalization  
  - Outlier capping or removal  
- **Reflect**  
  - Before/after metrics: Δ missing, Δ outliers, Δ skewness  
  - Historical logging for continuous policy refinement  
- **Rank**  
  - Random Forest feature-importance chart  
- **Explain**  
  - Jinja2-generated HTML report with embedded plots and tables  
  - Gradio UI for parameter tuning and one-click execution  

---

## 📐 Architecture Overview

```text
Raw Data → Data Loader
           ↓
       Profiling (Sense)
           ↓
Anomaly Flagging (Isolation Forest)
           ↓
  Policy Controller (Think)
           ↓
    Data Cleaning (Act)
           ↓
 Evaluation & Logging (Reflect)
           ↓
Feature Ranking + HTML Report

🗂 Repository Structure
c2c/
├── gradio_ui.py             # Gradio UI entrypoint
├── data_loader.py           # File ingestion (CSV/Excel/JSON)
├── data_profiler.py         # Statistical profiling & plots
├── anomaly_detector.py      # Isolation Forest & anomaly flags
├── adaptive_controller.py   # Policy reflection & adaptation
├── data_cleaner.py          # Imputation, deduplication, outliers, fixes
├── cleaning_evaluator.py    # Before/after metric computation
├── feature_ranker.py        # Random Forest feature importance
├── report_generator.py      # Jinja2 HTML report assembly
├── templates/
│   └── report_template.html # HTML report template
├── plots/                   # Generated plot images
├── reports/                 # HTML reports
├── outputs/                 # CSV exports (cleaned & anomalies)
└── logs/                    # JSONL run logs

Happy cleaning! 🧼
