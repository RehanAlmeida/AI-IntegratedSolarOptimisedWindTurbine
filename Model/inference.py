import joblib
import numpy as np
import pandas as pd

# ✅ Load the saved model
model_path = "solar_wind_classifier.pkl"
model = joblib.load(model_path)

# ✅ Define column names for input data
feature_names = ['wind_speed', 'humidity', 'temperature', 'wind_direction']

# ✅ Function for making predictions
def predict_solar_wind(data):
    """
    Predicts whether the given environmental conditions are favorable for Solar or Wind power.

    - Input: List[List[wind_speed, humidity, temperature, wind_direction]] (Single or multiple samples)
    - Output: List of tuples (Prediction, Confidence Score)
    """
    try:
        # Convert input to DataFrame
        df_input = pd.DataFrame(data, columns=feature_names)

        # Get predictions and confidence scores
        predictions = model.predict(df_input)
        probabilities = model.predict_proba(df_input)  # Confidence scores

        # Convert predictions to labels and return results
        results = [("Wind" if pred == 1 else "Solar", round(prob.max(), 3)) for pred, prob in zip(predictions, probabilities)]
        return results

    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Example Usage (Single & Multi-Sample)

# wind_speed, humidity, temperature, wind_direction

sample_data = [[1.5, 60, 29, 180], [3.5, 50, 25, 200]]  # Multiple samples
results = predict_solar_wind(sample_data)

# ✅ Print Predictions
for i, (prediction, confidence) in enumerate(results):
    print(f"Sample {i+1}: Predicted Class: {prediction} (Confidence: {confidence})")
