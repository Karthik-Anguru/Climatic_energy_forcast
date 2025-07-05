import joblib
import numpy as np

# Load model
model = joblib.load("models/energy_model.pkl")

def predict_energy(hour, dayofweek, is_weekend, month):
    features = np.array([[hour, dayofweek, is_weekend, month]])
    prediction = model.predict(features)[0]
    return round(prediction, 2)

# Example usage (can be removed in production)
if __name__ == "__main__":
    pred = predict_energy(hour=14, dayofweek=3, is_weekend=0, month=7)
    print(f"Predicted Energy Usage: {pred} kWh")
