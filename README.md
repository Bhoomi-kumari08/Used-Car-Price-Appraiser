# 🚗 Used Car Price Appraiser

A machine learning web application that predicts the resale value of a used car based on its specifications and features. Built using Python, Scikit-learn, and Streamlit.

## 🌐 Live Demo

[**Try the Used Car Price Appraiser**](https://used-car-price-appraiser-bhoomi.streamlit.app/)

## 📌 Project Overview

The Used Car Price Appraiser helps estimate the resale price of a car using a trained machine learning model. Users can enter car details through an interactive interface and receive a predicted price.

## ✨ Features

- Predicts used car resale prices.
- Interactive interface built with Streamlit.
- Accepts multiple car specifications as input.
- Uses Target Encoding for categorical features.
- Uses a trained Random Forest Regressor.
- Provides a web-based interface accessible online.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Categorical Encoding:** Category Encoders
- **Model Persistence:** Joblib
- **Web Framework:** Streamlit

## ⚙️ How It Works

1. The user enters the car's details.
2. The application processes the input features.
3. The saved Target Encoder transforms the relevant categorical features.
4. The trained Random Forest model predicts the estimated resale price.
5. The predicted price is displayed in the application.

## 📂 Project Files

- `app.py` — Streamlit application.
- `used_car_price_model1.pkl` — Saved machine learning model.
- `target_encoder1.pkl` — Saved Target Encoder.
- `feature_columns1.pkl` — Saved feature column information.
- `car_details_v4.csv` — Car dataset.
- `requirements.txt` — Python dependencies.

*Ensure these filenames match the actual files in your repository.*

## ▶️ Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/bhoomi-kumari08/Used-Car-Price-Appraiser.git
cd Used-Car-Price-Appraiser
```

**2. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**3. Start the application**

```bash
streamlit run app.py
```

## 🎯 Learning Outcomes

- Understanding supervised machine learning.
- Training and using a Random Forest regression model.
- Performing categorical feature encoding.
- Saving and loading trained models with Joblib.
- Deploying a machine learning application using Streamlit.

## 👩‍💻 Author

**Bhoomi Kumari**

B.Tech — Computer Science and Engineering