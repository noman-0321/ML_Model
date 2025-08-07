import joblib
import pandas as pd
import numpy as np

model = joblib.load("medical_rf_model.pkl")
new_data = pd.DataFrame([{
    "age": 25,
    "sex": 1,
    "bp": 110,
    "chol": 140,
    "fbs": 0,
    "restecg": 0,
    "exng": 0,
    "temperature": 37.5,
    "o2": -999,
    "hr": -999
}])

prediction = model.predict(new_data)
print(f"Predicted output: {prediction[0]}")  # 0 or 1
