# 📺 Content Monetization Modeler — YouTube Ad Revenue Predictor & Optimization Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Status](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)](tests/)
[![Docker Support](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

An end-to-end Machine Learning web application and analytics platform designed to predict, decompose, and optimize YouTube advertisement revenue (`ad_revenue_usd`) from viewer engagement and watch behavior metrics.

---

## 📌 Table of Contents
- [✨ Key Features](#-key-features)
- [🧠 Model Analysis & Scientific Findings](#-model-analysis--scientific-findings)
  - [Discovered Ground-Truth Equation](#discovered-ground-truth-equation)
  - [Root Cause Analysis: Why Previous Models Had High Error](#root-cause-analysis-why-previous-models-had-high-error)
  - [Benchmark Model Comparison](#benchmark-model-comparison)
- [🖥️ Streamlit Web Application Tour](#️-streamlit-web-application-tour)
- [🚀 Quickstart & Local Setup](#-quickstart--local-setup)
- [🧪 Running Tests](#-running-tests)
- [🚢 Production Deployment Guides](#-production-deployment-guides)
  - [Option 1: Deploy to Streamlit Community Cloud (Free & Recommended)](#option-1-deploy-to-streamlit-community-cloud-free--recommended)
  - [Option 2: Deploy with Docker](#option-2-deploy-with-docker)
  - [Option 3: Deploy to Render / Railway](#option-3-deploy-to-render--railway)
  - [Option 4: Deploy to Google Cloud Run](#option-4-deploy-to-google-cloud-run)
- [📁 Repository Structure](#-repository-structure)
- [👤 Author & Acknowledgments](#-author--acknowledgments)

---

## ✨ Key Features

1. **Precision Revenue Forecasting**:
   - Predicts ad earnings with zero boundary violations ($\text{Revenue} \ge \$0.00$).
   - Computes key creator metrics: **RPM (Revenue Per Mille)**, **AVD (Average View Duration)**, **Retention Rate %**, and **Engagement Rate %**.
2. **Component Revenue Decomposition**:
   - Interactive Plotly Waterfall and Donut charts quantifying the exact dollar contribution of **Base Views**, **Audience Engagement** (Likes + Comments), and **Watch Time**.
3. **Goal Seek & Reverse Optimization**:
   - Input your monetary goal (e.g., earn $500 or $1,000 on a video), and the engine calculates the required views, watch time, or likes/comments to bridge the gap.
   - Includes tactical creator playbooks (mid-roll ad placement timing, first 30s retention hooks).
4. **Batch Channel CSV Analysis**:
   - Drag-and-drop your YouTube Studio video export or test with our bundled 25-video demo dataset.
   - Computes channel-wide revenue projections, average RPM, scatter visualizations, and downloadable enriched CSVs.
5. **Model Diagnostics & Sensitivity Simulator**:
   - Real-time what-if sensitivity sliders to measure the marginal return of +1,000 views, +1,000 watch minutes, or +1,000 likes.

---

## 🧠 Model Analysis & Scientific Findings

### Discovered Ground-Truth Equation
During detailed regression analysis of the underlying 122,400 video records, we uncovered the exact structural generative formula:

$$\text{Ad Revenue (USD)} = 0.00521039 \cdot \text{Views} + 0.01718735 \cdot (\text{Likes} + \text{Comments}) + 0.00471454 \cdot \text{Watch Time (Minutes)}$$

#### Economic Interpretation:
- **Base Views**: **\$5.21 per 1,000 Views** (corresponds to programmatic base CPM).
- **Engagement Bonus**: **\$17.19 per 1,000 Engagements** (high like/comment volume signals active engagement, attracting higher advertiser auction bids).
- **Watch Time**: **\$4.71 per 1,000 Minutes Watched** (longer sessions trigger more ad impressions and mid-rolls).

---

### Root Cause Analysis: Why Previous Models Had High Error

In the original iteration, the legacy Lasso regression reported a test **MAE of \$3.08** and **RMSE of \$13.49**, while predicting negative revenue (**-\$13.92**) for low or zero inputs. Our empirical investigation uncovered the three root causes:

1. **Flawed Groupby Mean Imputation**:
   - Around 14.2% of the dataset (17,445 rows) had missing values for `likes`, `comments`, or `watch_time_minutes`.
   - The initial pipeline imputed missing values using group means across `country`, `device`, and `category`.
   - However, those categorical columns were uniformly distributed noise with virtually 0 correlation to watch time.
   - Imputing `watch_time_minutes` with the global mean (~37,500 min) for videos with true watch times of 15,000 min introduced artificial discrepancies of up to **\$56 on individual rows**, manufacturing high error strictly from data corruption.
   - **Validation Proof**:
     - MAE on uncorrupted rows: **\$0.119**
     - MAE on imputed rows: **\$22.33**
2. **Unscaled L1 Regularization Penalty**:
   - `Lasso(alpha=0.001)` was trained on unscaled features where `views` is in the thousands while ratio features like `engagement_rate` are near 0.1.
   - The L1 penalty zeroed out `engagement_rate` and biased the intercept to **-\$14.81**, causing predictions to diverge as view counts scaled up.
3. **Absence of Domain Boundary Constraints**:
   - Video revenue is strictly non-negative ($\ge \$0$). Unconstrained linear models produced negative revenue estimates for small channels.

---

### Benchmark Model Comparison

All models were evaluated on an identical 80/20 train/test holdout split:

| Model / Algorithm | Test R² | Test MAE ($) | Test RMSE ($) | Boundary Safe ($\ge \$0$) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Calibrated Non-Negative Regression (Production)** | **1.00000** | **0.0001** | **0.0001** | **✅ Yes** | **< 0.1 ms** |
| Standard Linear Regression | 1.00000 | 0.0000 | 0.0000 | ⚠️ No | < 0.1 ms |
| Scaled RidgeCV (Standardized) | 1.00000 | 0.0007 | 0.0011 | ⚠️ No | 0.2 ms |
| Scaled LassoCV (Standardized) | 0.99989 | 0.5236 | 0.6525 | ⚠️ No | 0.2 ms |
| HistGradientBoosting | 0.99992 | 0.4414 | 0.5610 | ✅ Yes | 1.8 ms |
| Random Forest (50 Trees) | 0.99983 | 0.6405 | 0.8149 | ✅ Yes | 12.4 ms |
| *Legacy Lasso (Unscaled + Corrupted Imputation)* | *0.95248* | *3.0820* | *13.4900* | *❌ No (-\$13.92)* | *0.2 ms* |

---

## 🖥️ Streamlit Web Application Tour

The web application is structured into four intuitive workspaces:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    📺 YouTube Ad Revenue Modeler                        │
├─────────────────────────────────────────────────────────────────────────┤
│ [ Tab 1: Single Video ] [ Tab 2: Goal Seek ] [ Tab 3: Batch ] [ Tab 4 ] │
│                                                                         │
│   ┌─────────────────────┐   ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │
│   │ Estimated Revenue   │   │   RPM   │ │Retention│ │ Engagement Rate │ │
│   │     $1,248.50       │   │  $12.48 │ │  62.5%  │ │      8.4%       │ │
│   └─────────────────────┘   └─────────┘ └─────────┘ └─────────────────┘ │
│                                                                         │
│   ┌───────────────────────────────┐   ┌─────────────────────────────┐   │
│   │   Attribution Waterfall       │   │     Revenue Share Donut     │   │
│   │   [Views | Eng | Watch Time]  │   │     [85% Watch | 11% View]  │   │
│   └───────────────────────────────┘   └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

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

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Tests

A test suite verifying model file integrity, zero-division safeguards, edge cases, and prediction accuracy is included:

```bash
python tests/run_tests.py
```
Or using pytest:
```bash
pytest tests/ -v
```

---

## 🚢 Production Deployment Guides

### Option 1: Deploy to Streamlit Community Cloud (Free & Recommended)
1. Fork or push this repository to your GitHub account: `https://github.com/your-username/content_monetization_modeler`.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New app"**.
4. Select:
   - **Repository**: `your-username/content_monetization_modeler`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**. Your application will be live in 1–2 minutes at a custom URL.

---

### Option 2: Deploy with Docker

A production multi-stage, non-root `Dockerfile` is included in the repository.

1. **Build the Docker Image**:
   ```bash
   docker build -t content-monetization-modeler .
   ```

2. **Run the Container**:
   ```bash
   docker run -d -p 8501:8501 --name youtube-revenue-app content-monetization-modeler
   ```
3. Visit `http://localhost:8501`.

---

### Option 3: Deploy to Render / Railway
This repository includes a `Procfile` and `runtime.txt`:
1. Connect your GitHub repo to [Render](https://render.com) or [Railway](https://railway.app).
2. Choose **Web Service**.
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

---

### Option 4: Deploy to Google Cloud Run
```bash
# Authenticate and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build & Deploy directly from source
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
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI test pipeline
├── .streamlit/
│   └── config.toml              # Streamlit server and dark-theme config
├── models/
│   ├── revenue_model.joblib     # Calibrated production model (joblib)
│   ├── feature_columns.joblib   # Feature columns schema
│   └── model_metadata.json      # Model provenance, metrics, and coefficients
├── tests/
│   ├── run_tests.py             # Test runner executable without external tools
│   ├── test_model.py            # Unit tests for model boundaries and serialization
│   └── test_app.py              # Unit tests for calculation logic and zero protection
├── 01_data_cleaning.ipynb       # Data cleaning and feature engineering
├── 02_eda.ipynb                 # Exploratory data analysis and correlations
├── 03_model_training.ipynb      # Rigorous model benchmarking and diagnostics
├── app.py                       # Main Streamlit web application
├── model_engine.py              # Core predictive math and batch processing engine
├── model.pkl                    # Calibrated model for backward compatibility
├── columns.pkl                  # Column index for backward compatibility
├── sample_channel_videos.csv    # Sample dataset for one-click batch demo
├── requirements.txt             # Pinned production dependencies
├── Dockerfile                   # Hardened multi-stage container
├── .dockerignore                # Exclusions for clean container builds
├── Procfile                     # PaaS process configuration
├── runtime.txt                  # Python runtime specification
└── README.md                    # Comprehensive documentation and deployment guide
```

---

## 👤 Author & Acknowledgments
- **Author**: Gopinath S
- **Repository**: [gopinath-sara1n/content_monetization_modeler](https://github.com/gopinath-sara1n/content_monetization_modeler)
- **License**: MIT
