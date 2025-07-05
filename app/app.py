import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
from src.recommend import get_recommendation

# ---------- Load Model ----------
try:
    model = joblib.load("models/energy_model.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ---------- Page Config ----------
st.set_page_config(page_title="Smart Energy Dashboard", page_icon="⚡", layout="wide")

# ---------- Header ----------
st.title("⚡ Smart Energy Usage Dashboard")
st.markdown("Predict household energy usage and explore patterns through interactive charts. 🌍")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📘 Input Parameters")
    hour = st.slider("Hour of Day", 0, 23, 14)
    day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    is_weekend = st.checkbox("Is it Weekend?", value=False)
    month = st.selectbox("Month", list(range(1, 13)), index=6)

# ---------- Predict ----------
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}
day_num = day_map[day]

features = np.array([[hour, day_num, is_weekend, month]])
prediction = round(model.predict(features)[0], 2)

# ---------- Results ----------
st.metric("🔋 Predicted Energy Usage", f"{prediction} kWh")
st.info(get_recommendation(prediction))

# ---------- Graph 1: Hourly Usage Forecast for Selected Day ----------
hourly_usage = pd.DataFrame({
    'Hour': list(range(24)),
    'Predicted Usage': [round(model.predict(np.array([[h, day_num, is_weekend, month]]))[0], 2) for h in range(24)]
})
fig_hour = px.line(hourly_usage, x='Hour', y='Predicted Usage', title="🔁 Hourly Energy Forecast", markers=True)
st.plotly_chart(fig_hour, use_container_width=True)

# ---------- Graph 2: Weekday vs Weekend Comparison ----------
usage_weekday = round(model.predict(np.array([[hour, day_num, False, month]]))[0], 2)
usage_weekend = round(model.predict(np.array([[hour, day_num, True, month]]))[0], 2)

comparison_df = pd.DataFrame({
    'Type': ['Weekday', 'Weekend'],
    'Predicted Usage': [usage_weekday, usage_weekend]
})
fig_compare = px.bar(comparison_df, x='Type', y='Predicted Usage', color='Type', title="📊 Weekday vs Weekend Comparison")
st.plotly_chart(fig_compare, use_container_width=True)

# ---------- Graph 3: Month-wise Forecast (at fixed hour & day) ----------
month_wise = pd.DataFrame({
    'Month': list(range(1, 13)),
    'Predicted Usage': [round(model.predict(np.array([[hour, day_num, is_weekend, m]]))[0], 2) for m in range(1, 13)]
})
fig_month = px.area(month_wise, x='Month', y='Predicted Usage', title="📆 Monthly Forecast Overview", markers=True)
st.plotly_chart(fig_month, use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("🔧 Built with ❤️ using Machine Learning, Plotly & Streamlit")
