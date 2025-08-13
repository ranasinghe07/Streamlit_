import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "boston_house_prices_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def train_and_save_model(df):
    y = df["MEDV"]
    X = df.drop(columns=["MEDV"]).select_dtypes(include=[np.number])
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.warning("Model not found — training a new one.")
        df_train = load_data()
        train_and_save_model(df_train)
        st.success("Model trained and saved.")
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

st.title("Boston Housing Price Predictor")
st.write("Predict `MEDV` based on housing features.")

page = st.sidebar.selectbox("Navigation", ["Home", "Data Exploration", "Predict"])

if page == "Home":
    st.write(df.describe())

elif page == "Data Exploration":
    col_x = st.selectbox("X-axis", df.columns, index=5)
    col_y = st.selectbox("Y-axis", df.columns, index=len(df.columns)-1)
    fig = px.scatter(df, x=col_x, y=col_y, color="CHAS")
    st.plotly_chart(fig)

elif page == "Predict":
    st.subheader("Enter Feature Values")
    inputs = {}
    for col in df.columns:
        if col != "MEDV":
            inputs[col] = st.number_input(col, value=float(df[col].mean()))
    if st.button("Predict"):
        X_new = pd.DataFrame([inputs])
        pred = model.predict(X_new)[0]
        st.success(f"Predicted MEDV: {pred:.2f}")
