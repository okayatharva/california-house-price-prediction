# 🏠 California House Price Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://california-house-price-predictor09.streamlit.app)

🚀 **Live Interactive App:** [Launch Web App](https://california-house-price-predictor09.streamlit.app)

An end-to-end Machine Learning pipeline and interactive Streamlit web application to predict California median home values using the California Housing dataset.

---

## 📌 Project Overview
* **Exploratory Data Analysis (EDA):** Visualized feature distributions and correlation heatmaps to understand property trends.
* **Modular Pipeline:** Built structured Python scripts for data loading (`load_data.py`), preprocessing (`preprocess.py`), and model training (`train_model.py`).
* **Model Benchmarking & Tuning:** Evaluated Linear Regression, Decision Trees, and Random Forest models.
* **Interactive Dashboard:** Deployed a real-time Streamlit web app allowing users to input district metrics and receive instant house price estimates.

---

## 🛠️ Tech Stack
* **Language:** Python
* **ML Libraries:** scikit-learn, pandas, numpy, joblib
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud

---

## 📁 Repository Structure
```text
california-house-price-prediction/
├── .github/
├── data/                  # Raw dataset files
├── models/                # Saved model binaries (.pkl)
├── notebooks/             # Exploratory notebooks & visual outputs
│   ├── 01_eda.ipynb
│   ├── housing_corr.png
│   └── housing_dist.png
├── src/                   # Source code
│   ├── load_data.py       # Data loading utility
│   ├── preprocess.py      # Feature scaling & train/test splitting
│   └── train_model.py     # Model training & artifact export
├── .gitignore             # Git exclusion rules
├── app.py                 # Interactive Streamlit application
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies