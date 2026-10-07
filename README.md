# 🏀 NBA Player Statistics Predictor

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![ML Framework](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready Machine Learning pipeline designed to forecast NBA player performance (Points, Rebounds, Assists) on a game-by-game basis. Built using domain-specific basketball analytics, rigorous time-series cross-validation, and state-of-the-art gradient boosting algorithms.

---

## 📌 Executive Summary

Predicting individual athletic performance in the NBA involves high variance due to dynamic match conditions, tactical changes, back-to-back schedules, and player fatigue. This project addresses the problem by constructing a predictive modeling engine that ingests historical box score data, builds temporal and opponent-adjusted features, and outputs probabilistic performance forecasts.

**Primary Business / Analytical Use Cases:**
* **Fantasy Sports Optimization:** Identify high-value target players based on predicted performance output.
* **Sports Analytics & Scouting:** Evaluate player efficiency trends against specific defensive schemes.
* **Predictive Benchmarking:** Establish quantitative baselines for expected player output under varying game contexts.

---

## ⚙️ Architecture & Data Pipeline

```
  ┌─────────────────┐       ┌────────────────────────┐       ┌───────────────────────────┐
  │  Data Sources   │ ────> │  Feature Engineering   │ ────> │  Time-Series Validation   │
  │ (nba_api / RAW) │       │ (Rolling Stats/Rest)   │       │    (Expanding Window)     │
  └─────────────────┘       └────────────────────────┘       └───────────────────────────┘
                                                                           │
                                                                           ▼
  ┌─────────────────┐       ┌────────────────────────┐       ┌───────────────────────────┐
  │   Model Output  │ <──── │ Model Interpretability │ <──── │   Gradient Boosting /     │
  │   & Dashboards  │       │     (SHAP Analysis)    │       │     Ensemble Models       │
  └─────────────────┘       └────────────────────────┘       └───────────────────────────┘
```

1. **Data Ingestion:** Automated extraction of historical box scores, team pace, defensive ratings, and rest-day variables using official API endpoints.
2. **Feature Engineering:** Creation of time-aware indicators without data leakage (e.g., $N$-game rolling averages, PER proxies, back-to-back indicators, matchup-specific defensive efficiency).
3. **Cross-Validation Strategy:** Custom **Expanding Window Cross-Validation** to prevent temporal data leakage and mirror real-world forecasting.
4. **Model Architecture:** Ensemble combining **XGBoost**, **LightGBM**, and **CatBoost** tuned via **Optuna**.
5. **Model Explainability:** SHAP (SHapley Additive exPlanations) values to interpret feature impacts on individual predictions.

---

## ✨ Key Features

- **Leakage-Free Time Series Design:** All features (rolling metrics, usage rates, trends) are strictly computed on historical games prior to the target match date.
- **Advanced Sports Metrics:**
  - Dynamic **Rolling Windows** (3, 5, 10, and 20-game horizons) for Points, Assists, Rebounds, Minutes, and Usage Rate.
  - **Opponent Defensive Adjustment:** Normalizes player output against opponent defensive ratings and matchup pace.
  - **Schedule Fatigue Matrix:** Quantifies impact of rest days, travel, and back-to-back game schedules.
- **Automated Hyperparameter Tuning:** Optuna framework integrated for automated Bayesian optimization.
- **Model Explainability Engine:** Built-in SHAP visualizers to explain *why* a model predicts a specific statistic for a given matchup.

---

## 📊 Evaluation Metrics & Benchmarks

Models are evaluated across key regression metrics: Mean Absolute Error ($MAE$), Root Mean Squared Error ($RMSE$), and Coefficient of Determination ($R^2$).

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### Sample Baseline Comparison (Points Prediction Target)

| Model Architecture | MAE (Points) | RMSE (Points) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| Baseline (5-Game Rolling Average) | 4.82 | 6.21 | 0.51 |
| Ridge Regression (L2) | 4.25 | 5.54 | 0.62 |
| Random Forest Regressor | 3.98 | 5.12 | 0.68 |
| **XGBoost (Tuned)** | **3.51** | **4.68** | **0.75** |
| **Ensemble (XGB + LightGBM)** | **3.42** | **4.55** | **0.77** |

---

---

## 🚀 Quickstart & Setup

### Prerequisites
* Python `3.10` or higher
* `pip` or `conda` environment manager

### 1. Clone the Repository
```bash
git clone https://github.com/Jawickz/nbaplayer-statspredictor.git
cd nbaplayer-statspredictor
```

### 2. Set Up Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Data Ingestion & Feature Engineering
```bash
python src/data_loader.py --season 2023-24
python src/features.py
```

### 5. Train Model & Generate Predictions
```bash
python src/train.py --target PTS --model xgboost
python src/predict.py --player "LeBron James" --date "latest"
```

---

## 🔮 Roadmap & Future Improvements

- [ ] **Player Status/Injury Integration:** Incorporate real-time injury report scrapers to dynamically update target minute projections.
- [ ] **Deep Learning Explorations:** Implement Temporal Fusion Transformers (TFT) or LSTM networks for complex sequential patterns.
- [ ] **Web Dashboard:** Build an interactive Streamlit/Dash application for real-time player matchup analysis.
- [ ] **Dockerization & CI/CD:** Package the inference pipeline into Docker containers with automated GitHub Actions testing.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the Repository.
2. Create a Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Author:** [Jawickz](https://github.com/Jawickz)
