import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
from src.recommend import get_recommendation

# ---------- Load Trained Model ----------
try:
    model = joblib.load("models/energy_model.pkl")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

# ---------- Page Config ----------
st.set_page_config(page_title="🌤️ Climatic Energy Forecast", page_icon="🔌", layout="centered")

# ---------- Header ----------
st.title("🌤️ Climatic Energy Forecast")
st.markdown("""
This AI-powered tool predicts household energy usage based on climatic factors like time of day, weekday, month, and weekend status.
Get personalized recommendations and visual insights to reduce energy consumption sustainably. 🌱
""")

# ---------- Sidebar Inputs ----------
with st.sidebar:
    st.header("🛠️ Forecast Settings")
    hour = st.slider("Hour of Day", 0, 23, 14)
    day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    is_weekend = st.checkbox("Is it a Weekend?", value=False)
    month = st.selectbox("Month", list(range(1, 13)), index=6)

# Map day name to number
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}
day_num = day_map[day]

# ---------- Predict Button ----------
if st.button("🔍 Forecast Energy Usage"):
    features = np.array([[hour, day_num, is_weekend, month]])
    prediction = round(model.predict(features)[0], 2)

    st.subheader("🔋 Energy Forecast Result")
    st.metric("Predicted Usage", f"{prediction} kWh")
    st.success(get_recommendation(prediction))

    # ---------- Chart 1: Hourly Forecast for Selected Day ----------
    hourly_usage = pd.DataFrame({
        'Hour': list(range(24)),
        'Predicted Usage': [round(model.predict(np.array([[h, day_num, is_weekend, month]]))[0], 2) for h in range(24)]
    })
    fig_hour = px.line(hourly_usage, x='Hour', y='Predicted Usage', title="📈 Hourly Energy Forecast", markers=True)
    st.plotly_chart(fig_hour, use_container_width=True)

    # ---------- Chart 2: Monthly Variation at Selected Hour ----------
    month_usage = pd.DataFrame({
        'Month': list(range(1, 13)),
        'Predicted Usage': [round(model.predict(np.array([[hour, day_num, is_weekend, m]]))[0], 2) for m in range(1, 13)]
    })
    fig_month = px.area(month_usage, x='Month', y='Predicted Usage', title="📆 Month-wise Usage Forecast")
    st.plotly_chart(fig_month, use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("Developed by **Karthik Anguru** | Powered by Streamlit, Plotly & Machine Learning ⚡")
