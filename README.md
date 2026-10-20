# 📺 Content Monetization Modeler — YouTube Ad Revenue Predictor & Optimization Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://contentmonetizationmodeler-mrm5xcpsx3pf3rvt4gwhd6.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Status](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)](tests/)
[![Docker Support](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

> 🌐 **Live Web Application**: [https://contentmonetizationmodeler-mrm5xcpsx3pf3rvt4gwhd6.streamlit.app/](https://contentmonetizationmodeler-mrm5xcpsx3pf3rvt4gwhd6.streamlit.app/)

An end-to-end Machine Learning web application and creator analytics platform designed to predict, decompose, and optimize YouTube advertisement revenue (`ad_revenue_usd`) from engagement and watch behavior metrics.

---

## 📌 Table of Contents
- [✨ Key Features](#-key-features)
- [🏆 Selected Final Model](#-selected-final-model)
- [🧠 Model Analysis & Scientific Findings](#-model-analysis--scientific-findings)
  - [Discovered Ground-Truth Equation](#discovered-ground-truth-equation)
  - [Root Cause Analysis: Why Previous Models Had High Error](#root-cause-analysis-why-previous-models-had-high-error)
  - [Benchmark Model Comparison](#benchmark-model-comparison)
- [🖥️ Streamlit Web Application Tour](#️-streamlit-web-application-tour)
- [🚀 Quickstart & Local Setup](#-quickstart--local-setup)
- [🧪 Running Tests](#-running-tests)
- [🚢 Production Deployment Guides](#-production-deployment-guides)
  - [Streamlit Community Cloud](#streamlit-community-cloud)
  - [Docker Container](#docker-container)
  - [PaaS (Render / Railway / Heroku)](#paas-render--railway--heroku)
  - [Google Cloud Run](#google-cloud-run)
- [📁 Repository Structure](#-repository-structure)
- [👤 Author & License](#-author--license)

---

## ✨ Key Features

1. **Precision Revenue Forecasting**:
   - Zero boundary violations: strictly non-negative ($\text{Revenue} \ge \$0.00$).
   - Computes key performance metrics: **RPM (Revenue Per Mille)**, **AVD (Average View Duration)**, **Retention Rate %**, and **Engagement Rate %**.
2. **Interactive Component Attribution**:
   - Plotly Waterfall and Donut charts quantifying exact dollar contributions from **Base Views**, **Audience Engagement** (Likes + Comments), and **Watch Time**.
3. **Goal Seek & Reverse Optimizer**:
   - Input any target revenue ($500, $1,000, $5,000) to reverse-calculate the exact views, watch minutes, or engagement required to hit your target.
   - Includes tactical playbooks (mid-roll ad placement timing, first-30s hook retention).
4. **Batch Channel CSV Analysis**:
   - Upload YouTube Studio exports or load the built-in 25-video demo dataset with one click.
   - Computes channel totals, average RPM, interactive scatter distributions, and downloadable enriched CSVs.
5. **Streamlined Model Selection & Diagnostics Dashboard**:
   - De-cluttered executive layout featuring the **Selected Final Model**, benchmark horizontal bar charts, error root-cause diagnosis, and marginal sensitivity what-if sliders.

---

## 🏆 Selected Final Model

The primary production engine selected for all final predictions is the **Calibrated Non-Negative Regressor (v2.0)**:

| Metric | Score / Value | Benefit |
| :--- | :---: | :--- |
| **Test R²** | **1.00000** | 100% of revenue variance captured |
| **Test MAE** | **$0.0001** | Zero prediction loss (machine precision) |
| **Test RMSE** | **$0.0001** | Eliminates severe residual spikes |
| **Boundary Safety** | **Strictly $\ge \$0.00$** | Cannot output negative revenue |
| **Inference Latency** | **< 0.1 ms** | 120x faster than Random Forest |
| **Explainability** | **100% Transparent** | Exact linear component attribution |

### Selection Rationale:
- **Eliminates Intercept Distortion**: Legacy Lasso models had an intercept of **-$14.81**, causing zero-metric videos to predict negative earnings. The calibrated model forces non-negative weights and an exact zero baseline.
- **Resilient to Imputation Corruption**: Unaffected by the 14.2% missing-value imputation noise that contaminated previous models.
- **Physical Domain Alignment**: Ad revenue cannot mathematically fall below zero.

---

## 🧠 Model Analysis & Scientific Findings

### Discovered Ground-Truth Equation
Across ~120,000 video records, we uncovered the exact generative monetization formula:

$$\text{Ad Revenue (USD)} = 0.00521039 \cdot \text{Views} + 0.01718735 \cdot (\text{Likes} + \text{Comments}) + 0.00471454 \cdot \text{Watch Time (Minutes)}$$

#### Economic Breakdown:
- **Base Views**: **\$5.21 per 1,000 Views** (programmatic base CPM impression rate).
- **Engagement Premium**: **\$17.19 per 1,000 Engagements** (auction bonus triggered by high like/comment velocity).
- **Watch Time Multiplier**: **\$4.71 per 1,000 Minutes Watched** (mid-roll impressions and extended session depth).

---

### Root Cause Analysis: Why Previous Models Had High Error

The legacy repository reported an MAE of **\$3.08** and RMSE of **\$13.49** for Lasso regression. Our audit diagnosed the three root causes:

1. **Flawed Groupby Mean Imputation**:
   - 14.2% of rows (17,445 records) had missing values for likes, comments, or watch time.
   - The initial pipeline imputed missing values using group means grouped by `['country', 'device', 'category']`.
   - However, those three categorical columns were uniformly distributed noise with **zero correlation** to watch time.
   - Imputing `watch_time_minutes` with the dataset mean (~37,500 min) for videos with true watch times of 15,000 min introduced errors up to **\$56 on individual rows**.
   - **Empirical Validation**:
     - Test MAE on complete rows: **\$0.119**
     - Test MAE on imputed rows: **\$22.33** (responsible for 99% of legacy error!)
2. **Unscaled L1 Regularization Penalty**:
   - Legacy Lasso was trained on unscaled features where `views` is in the thousands while ratio features like `engagement_rate` are ~0.1.
   - The L1 penalty zeroed out `engagement_rate` and biased the intercept to **-\$14.81**.
3. **Domain Boundary Violations**:
   - Unconstrained models predicted **-\$13.92** for videos with low view counts.

---

### Benchmark Model Comparison

All candidate models were evaluated on an identical 80/20 train/test holdout split:

| Model / Algorithm | Test R² | Test MAE ($) | Test RMSE ($) | Boundary Safe ($\ge \$0$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Calibrated Non-Negative (Production)** | **1.00000** | **0.0001** | **0.0001** | **✅ Yes** | **🏆 Selected Final Model** |
| Standard Linear Regression | 1.00000 | 0.0000 | 0.0000 | ⚠️ No | Unconstrained |
| Scaled RidgeCV (Standardized) | 1.00000 | 0.0007 | 0.0011 | ⚠️ No | Regularized |
| HistGradientBoosting | 0.99992 | 0.4414 | 0.5610 | ✅ Yes | Tree Ensemble |
| Scaled LassoCV (Standardized) | 0.99989 | 0.5236 | 0.6525 | ⚠️ No | Over-shrunk |
| Random Forest (50 Trees) | 0.99983 | 0.6405 | 0.8149 | ✅ Yes | High Latency |
| *Legacy Lasso (Unscaled + Imputed)* | *0.95248* | *3.0820* | *13.4900* | *❌ No (-\$13.92)* | *Defective (Legacy)* |

---

## 🖥️ Streamlit Web Application Tour

The web application features four dedicated tabs:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    📺 YouTube Ad Revenue Modeler                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ [ Tab 1: Single Video ] [ Tab 2: Goal Seek ] [ Tab 3: Batch ] [ Tab 4: Model ]│
│                                                                              │
│   ┌─────────────────────┐   ┌─────────┐ ┌─────────┐ ┌──────────────────────┐ │
│   │ Estimated Revenue   │   │   RPM   │ │Retention│ │   Engagement Rate    │ │
│   │     $1,248.50       │   │  $12.48 │ │  62.5%  │ │        8.4%          │ │
│   └─────────────────────┘   └─────────┘ └─────────┘ └──────────────────────┘ │
│                                                                              │
│   ┌────────────────────────────────┐   ┌──────────────────────────────────┐  │
│   │   Attribution Waterfall        │   │       Revenue Share Donut        │  │
│   │   [Views | Eng | Watch Time]   │   │       [85% Watch | 11% View]     │  │
│   └────────────────────────────────┘   └──────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **Tab 1: 🔮 Video Revenue Predictor**: Real-time revenue estimation, RPM, retention %, average view duration (AVD), and interactive Plotly waterfall decomposition.
- **Tab 2: 🎯 Goal Seek & Optimizer**: Input a revenue goal to reverse-solve the required views, watch time, or engagement, paired with tactical creator advice.
- **Tab 3: 📁 Batch Channel Predictor**: Drag-and-drop CSV upload with a 1-click **"Load Demo Channel Dataset"** button and enriched CSV export.
- **Tab 4: 🔬 Model Selection & Diagnostics**: Streamlined executive section featuring the selected model, interactive benchmark bar chart, error diagnosis, and what-if sensitivity sliders.

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/gopinath-sara1n/content_monetization_modeler.git
cd content_monetization_modeler
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Or simply:
```bash
python app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Tests

A comprehensive unit test suite verifies model integrity, zero-division protection, and calculation accuracy:

```bash
python tests/run_tests.py
```

---

## 🚢 Production Deployment Guides

### Streamlit Community Cloud
Live link: **[https://contentmonetizationmodeler-mrm5xcpsx3pf3rvt4gwhd6.streamlit.app/](https://contentmonetizationmodeler-mrm5xcpsx3pf3rvt4gwhd6.streamlit.app/)**

To deploy your own fork:
1. Log in to [share.streamlit.io](https://share.streamlit.io).
2. Click **"New app"**.
3. Select repository: `your-username/content_monetization_modeler`, branch: `main`, file: `app.py`.
4. Click **"Deploy!"**. The included `.streamlit/config.toml` and `.streamlit/credentials.toml` ensure smooth, unattended cloud execution.

### Docker Container
```bash
# Build
docker build -t content-monetization-modeler .

# Run
docker run -d -p 8501:8501 --name youtube-revenue-app content-monetization-modeler
```
Visit `http://localhost:8501`.

### PaaS (Render / Railway / Heroku)
Uses the included `Procfile` and `runtime.txt`:
```bash
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Google Cloud Run
```bash
gcloud run deploy content-monetization-modeler \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501
```

---

## 📁 Repository Structure

```
content_monetization_modeler/
├── .devcontainer/
│   └── devcontainer.json        # VS Code remote development config
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI test pipeline
├── .streamlit/
│   ├── config.toml              # Streamlit server and dark-theme config
│   └── credentials.toml         # Automated cloud onboarding bypass
├── models/
│   ├── revenue_model.joblib     # Production calibrated model
│   ├── feature_columns.joblib   # Schema definition
│   └── model_metadata.json      # Provenance, metrics, and parameters
├── tests/
│   ├── run_tests.py             # Test suite runner
│   ├── test_model.py            # Unit tests for model boundaries
│   └── test_app.py              # Unit tests for calculation engine
├── 01_data_cleaning.ipynb       # Data preprocessing and feature engineering
├── 02_eda.ipynb                 # Correlation and distribution analysis
├── 03_model_training.ipynb      # Rigorous model benchmarking and diagnostics
├── app.py                       # Main Streamlit web application
├── model_engine.py              # Core predictive math and batch processing engine
├── model.pkl                    # Calibrated model for backward compatibility
├── columns.pkl                  # Column index for backward compatibility
├── sample_channel_videos.csv    # Sample dataset for one-click batch demo
├── requirements.txt             # Pinned production dependencies
├── Dockerfile                   # Hardened multi-stage container
├── .dockerignore                # Container build exclusions
├── Procfile                     # PaaS process configuration
├── runtime.txt                  # Python runtime specification
└── README.md                    # Project documentation and deployment guide
```

---

## 👤 Author & License
- **Author**: Gopinath S
- **Repository**: [gopinath-sara1n/content_monetization_modeler](https://github.com/gopinath-sara1n/content_monetization_modeler)
- **Live App**: [contentmonetizationmodeler-mrm5xcpsx3pf3rvt4gwhd6.streamlit.app](https://contentmonetizationmodeler-mrm5xcpsx3pf3rvt4gwhd6.streamlit.app/)
- **License**: MIT
