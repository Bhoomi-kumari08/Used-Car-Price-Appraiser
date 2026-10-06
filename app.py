import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "used_car_price_model1.pkl")
encoder = joblib.load(BASE_DIR / "models" / "target_encoder1.pkl")
feature_columns = joblib.load(BASE_DIR / "models" / "feature_columns1.pkl")

df = pd.read_csv(BASE_DIR / "data" / "car details v4.csv")

st.set_page_config(
    page_title="Used Car Price Appraiser",
    page_icon="🚗",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>🚗 Used Car Price Appraiser</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
    "<h5 style='text-align: center; color: gray;'>Predict the resale value of your used car using a Machine Learning model.</h5>",
    unsafe_allow_html=True
)

st.header("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox(
        "Make",
        sorted(df["Make"].dropna().unique())
    )
    model_name = st.selectbox(
        "Model",
        sorted(df["Model"].dropna().unique())
    )
    year = st.number_input(
        "Manufacturing Year",
        1990,
        2026,
        2020
    )
    kilometer = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=50000
    )
    fuel = st.selectbox(
        "Fuel Type",
        sorted(df["Fuel Type"].dropna().unique())
    )
    transmission = st.selectbox(
        "Transmission",
        sorted(df["Transmission"].dropna().unique())
    )
    location = st.selectbox(
        "Location",
        sorted(df["Location"].dropna().unique())
    )
with col2:
    color = st.selectbox(
        "Color",
        sorted(df["Color"].dropna().unique())
    )
    owner = st.selectbox(
        "Owner",
        sorted(df["Owner"].dropna().unique())
    )
    seller = st.selectbox(
        "Seller Type",
        sorted(df["Seller Type"].dropna().unique())
    )
    engine = st.number_input(
        "Engine (cc)",
        value=1200
    )
    power = st.number_input(
        "Max Power (bhp)",
        value=80.0
    )
    torque = st.number_input(
        "Max Torque (Nm)",
        value=110.0
    )
    drivetrain = st.selectbox(
        "Drivetrain",
        sorted(df["Drivetrain"].dropna().unique())
    )
if st.button("Predict Price"):
    from datetime import datetime
    car_age = datetime.now().year - year

    input_df = pd.DataFrame({
        "Make": [make],
        "Model": [model_name],
        "Kilometer": [kilometer],
        "Fuel Type": [fuel],
        "Transmission": [transmission],
        "Location": [location],
        "Color": [color],
        "Owner": [owner],
        "Seller Type": [seller],
        "Engine": [engine],
        "Max Power": [power],
        "Max Torque": [torque],
        "Drivetrain": [drivetrain],
        "Length": [df["Length"].median()],
        "Width": [df["Width"].median()],
        "Height": [df["Height"].median()],
        "Seating Capacity": [df["Seating Capacity"].median()],
        "Fuel Tank Capacity": [df["Fuel Tank Capacity"].median()],
        "Car Age": [car_age]
    })

    input_df = encoder.transform(input_df)

    input_df = pd.get_dummies(
        input_df,
        columns=[
            "Make",
            "Fuel Type",
            "Transmission",
            "Color",
            "Owner",
            "Seller Type",
            "Drivetrain"
        ],
        drop_first=True
    )

    bool_cols = input_df.select_dtypes(include="bool").columns
    input_df[bool_cols] = input_df[bool_cols].astype(int)

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_df)

    st.success(
        f"Estimated Used Car Price: ₹ {prediction[0]:,.0f}"
    )

st.markdown("---")

st.markdown(
    "<p style='text-align:center;color:gray;'>Made with ❤️ using Streamlit & Scikit-learn</p>",
    unsafe_allow_html=True
)