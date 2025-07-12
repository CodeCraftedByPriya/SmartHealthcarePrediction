from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from predict import get_aggregated_insights, SATDoc_chart, RecHos_chart
import base64

app = Flask(__name__)

# Load trained models and scaler
clf_model = joblib.load("diagnosis_model.pkl")
reg_model = joblib.load("recovery_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the order of input features
feature_order = [
    'Age', 'Gender', 'Heart_Rate', 'Temperature', 'Systolic', 'Diastolic',
    'X-ray_Results', 'Lab_Test_Results', 'FamilyHistory', 'Allergies',
    'Hypertension_Risk'
]

model_features = joblib.load("model_features.pkl")


@app.route('/')
def index():
    return render_template("home.html")


chart1 = None
chart2 = None


@app.route('/analytics')
def analytics():
    global chart1, chart2
    if chart1 is None or chart2 is None:
        chart1 = SATDoc_chart()
        chart2 = RecHos_chart()
    insights = get_aggregated_insights()
    return render_template('analytics.html', chart1=chart1, chart2=chart2, insights=insights)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form

        # Convert and validate inputs
        input_data = {
            'Age': int(data['age']),
            'Gender': int(data['gender']),
            'Heart_Rate': float(data['heart_rate']),
            'Temperature': float(data['temperature']),
            'Systolic': float(data['systolic']),
            'Diastolic': float(data['diastolic']),
            'X-ray_Results': int(data['xray']),
            'Lab_Test_Results': float(data['lab']),
            'FamilyHistory': int(data['family_history']),
            'Allergies': int(data['allergies'])
        }

        # Derived feature
        input_data['Hypertension_Risk'] = int(
            input_data['Systolic'] >= 130 or input_data['Diastolic'] >= 80
        )

        # Create input DataFrame with all possible input features
        input_df = pd.DataFrame([input_data])

        # Keep only the columns used in training (and in the correct order)
        input_df = input_df[model_features]

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Make predictions
        diagnosis = clf_model.predict(scaled_input)[0]
        recovery_days = reg_model.predict(scaled_input)[0]

        return render_template(
            "home.html",
            diagnosis=diagnosis,
            recovery=round(recovery_days, 1),
            input=input_data
        )

    except Exception as e:
        return render_template("home.html", error=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
