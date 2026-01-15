# 🏆 Gold Price Forecasting Using AI & Machine Learning

**Comparative Analysis: Prophet vs SARIMAX vs LightGBM**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

This project implements and compares **three advanced AI/ML models** for forecasting gold prices:
- **Prophet** (Meta's time series forecasting)
- **SARIMAX** (Statistical modeling)
- **LightGBM** (Gradient boosting machine learning)

**Objective:** Predict gold futures prices with high accuracy to support investment decisions.

---

## 🎯 Key Features

✅ **Real-time data** fetched from Yahoo Finance (2020-2026)  
✅ **3 model comparison** with comprehensive evaluation metrics  
✅ **60-day price forecasts** with confidence intervals  
✅ **Interactive visualizations** using Plotly and Matplotlib  
✅ **Technical indicators**: RSI, MACD, Moving Averages  
✅ **Professional dashboards** with trading signals  

---

## 📊 Model Performance

| Model | MAPE | RMSE | MAE | R² Score |
|-------|------|------|-----|----------|
| **Prophet** | **11.53%** | $560.57 | $429.06 | -0.066 |
| SARIMAX | 19.09% | $881.70 | $707.79 | -1.672 |
| LightGBM | 21.46% | $945.15 | $786.10 | -2.071 |

**Winner: Prophet** - Best balance of accuracy and reliability

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation

```bash
# Clone repository
git clone https://github.com/VarunTehlan/gold-price-forecasting-ai.git
cd gold-price-forecasting-ai

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Gold_Price_Forecasting_AI_Project.ipynb
```

---
## 📁 Project Structure

```bash
gold-price-forecasting-ai/
│
├── Gold_Price_Forecasting_AI_Project.ipynb  # Main analysis notebook
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
└── .gitignore                                # Git ignore rules
```

---

## 🔬 Methodology

### 1. Data Collection
- **Source:** Yahoo Finance (GC=F - Gold Futures)
- **Period:** January 2020 - January 2026
- **Records:** 1,516 trading days
- **Features:** Open, High, Low, Close, Volume

### 2. Data Preprocessing
- Missing value imputation
- Outlier detection & treatment
- Feature engineering (technical indicators)
- Train-test split (80-20)

### 3. Model Training
- **Prophet:** Additive time series model with seasonality
- **SARIMAX:** (2,1,2)x(1,1,1,12) with trend component
- **LightGBM:** 500 estimators, max depth 7, learning rate 0.05

### 4. Evaluation Metrics
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

## 📈 Key Insights

🔹 **Prophet outperforms** with 11.46% MAPE  
🔹 **Gold showed high volatility** in 2024-2025 period  
🔹 **Model predicts correction** (-19% over 60 days)  
🔹 **RSI indicates neutral** momentum (64.08)  
🔹 **MACD signals bullish** short-term trend  

---

## 🛠️ Technologies Used

**Languages:** Python 3.8+

**Libraries:**
- Data: `pandas`, `numpy`, `yfinance`
- ML/AI: `prophet`, `statsmodels`, `lightgbm`, `scikit-learn`
- Visualization: `matplotlib`, `seaborn`, `plotly`

**Tools:** Jupyter Notebook, Git, GitHub

---

## 📝 Future Enhancements

- [ ] Add LSTM/GRU deep learning models
- [ ] Implement real-time prediction API
- [ ] Include sentiment analysis from news
- [ ] Deploy as web application (Streamlit)
- [ ] Add cryptocurrency correlation analysis

---

## 👤 Author

**Varun Tehlan**  
Business/Data Analyst | AI/ML Enthusiast

📧 Email: varun.tehlan@gmail.com  
💼 LinkedIn: [linkedin.com/in/varuntehlan](https://www.linkedin.com/in/varuntehlan)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Meta's Prophet library for time series forecasting
- Yahoo Finance for real-time market data
- Scikit-learn community for ML tools

---

**⭐ Star this repository if you find it helpful!**
