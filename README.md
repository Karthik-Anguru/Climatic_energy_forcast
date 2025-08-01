﻿# Climatic_energy_forcast
⚡ Smart Energy Usage Forecasting
A Streamlit-based web application that predicts household energy consumption using time and calendar inputs. It also provides actionable energy-saving recommendations and interactive visualizations.

🧠 Features
- 🔮 Predict energy usage based on:
  - Hour of the day
  - Day of the week
  - Month
  - Weekend status
- 📈 Visualizations:
  - Hourly energy trends
  - Weekday vs Weekend comparison
  - Month-wise consumption
- 💡 Dynamic energy efficiency recommendations
- 🎨 Custom CSS for a polished UI
- ⚙️ Powered by Machine Learning (Random Forest)

📁 Project Structure

climatic-energy-forecast/
├── app/                     # Streamlit UI
│   └── app.py
├── data/                    # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── models/                  # Trained model files (.pkl)
├── notebooks/               # Jupyter notebooks for EDA and training
├── reports/                 # Charts and EDA summaries
├── src/                     # Core Python scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── recommend.py
│   └── feature_engineering.py
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignored files
└── README.md

🚀 How to Run Locally

1. Clone the Repository
   git clone https://github.com/your-username/climatic-energy-forecast.git
   cd climatic-energy-forecast

2. Create a Virtual Environment
   python -m venv venv
   # Activate:
   # Windows: venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate

3. Install Dependencies
   pip install -r requirements.txt

4. Run the App
   streamlit run app/app.py

📦 Requirements

- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- plotly

Install with:
pip install -r requirements.txt

📷 Screenshots
Add your screenshots to `/screenshots/` and reference them in your README as needed.
✍️ Author

This project is licensed under the MIT License - see the LICENSE file for details.
📝 Acknowledgments

- UCI Household Power Consumption Dataset
- Streamlit for the frontend
- Plotly for visualization
