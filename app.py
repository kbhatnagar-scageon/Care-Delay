from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="Hospital Care Delay Prediction API",
    description="""
    Predicts care delay for patients based on ward conditions and patient attributes.
    
    ## Usage Example
    
    ```bash
    curl -X 'POST' \\
      'http://localhost:8000/predict' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "age": 65,
        "gender": "M",
        "num_comorbidities": 2,
        "severity_score": 3,
        "primary_diagnosis": "Pneumonia",
        "registration_timestamp": "2025-05-15 10:30:00",
        "total_beds": 30,
        "occupied_beds": 25,
        "staff_count": 5,
        "staff_to_patient_ratio": 5.0
    }'
    ```
    """,
    version="1.1.0",
)

# Load the model and required files
model_path = "care_delay_prediction_model.pkl"
features_path = "model_features.pkl"
model_info_path = "model_info.pkl"

# Check if model files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file {model_path} not found. Please train the model first."
    )

if not os.path.exists(features_path):
    raise FileNotFoundError(
        f"Features file {features_path} not found. Please train the model first."
    )

if not os.path.exists(model_info_path):
    raise FileNotFoundError(
        f"Model info file {model_info_path} not found. Please train the model first."
    )

# Load model and features
with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(features_path, "rb") as file:
    required_features = pickle.load(file)

with open(model_info_path, "rb") as file:
    model_info = pickle.load(file)


# Define Pydantic models for request and response
class PatientData(BaseModel):
    age: int = Field(..., example=65, description="Patient age in years")
    gender: str = Field(..., example="M", description="Patient gender (M/F)")
    num_comorbidities: int = Field(
        ..., example=2, description="Number of comorbidities"
    )
    severity_score: int = Field(..., example=3, description="Severity score (1-5)")
    primary_diagnosis: str = Field(
        ..., example="Pneumonia", description="Primary diagnosis"
    )

    registration_timestamp: str = Field(
        ...,
        example="2025-05-15 10:30:00",
        description="Registration timestamp (YYYY-MM-DD HH:MM:SS)",
    )

    total_beds: int = Field(..., example=30, description="Total beds in ward")
    occupied_beds: int = Field(..., example=25, description="Currently occupied beds")
    staff_count: int = Field(..., example=5, description="Number of staff members")
    staff_to_patient_ratio: float = Field(
        ..., example=5.0, description="Staff to patient ratio"
    )

    # Optional fields - will be calculated if not provided
    occupancy_rate: Optional[float] = Field(
        None, example=83.3, description="Ward occupancy rate (%)"
    )
    patients_per_staff: Optional[float] = Field(
        None, description="Number of patients per staff member"
    )
    comorbidity_severity: Optional[int] = Field(
        None, description="Interaction of comorbidities and severity"
    )


class PredictionResponse(BaseModel):
    predicted_care_delay: float = Field(
        ..., description="Predicted care delay in minutes"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Helper function to extract time features from timestamp
def extract_time_features(timestamp_str: str) -> dict:
    """Extract time-related features from a timestamp string"""
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError("Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS")

    # Extract basic time information
    hour_of_day = timestamp.hour
    day_of_week = timestamp.weekday() + 1  # 1=Monday, 7=Sunday

    # Derive time-based features
    is_weekend = 1 if day_of_week >= 6 else 0
    is_peak_morning = 1 if 6 <= hour_of_day < 9 else 0
    is_peak_afternoon = 1 if 13 <= hour_of_day < 15 else 0
    is_busy_day = 1 if day_of_week in [1, 5] else 0  # Monday or Friday

    # Determine shift
    if 7 <= hour_of_day < 15:
        shift = "Morning"
    elif 15 <= hour_of_day < 23:
        shift = "Evening"
    else:
        shift = "Night"

    # Cyclical encodings for hour and day
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
    day_sin = np.sin(2 * np.pi * (day_of_week - 1) / 7)
    day_cos = np.cos(2 * np.pi * (day_of_week - 1) / 7)

    # Return all time-related features
    return {
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "shift": shift,
        "is_weekend": is_weekend,
        "is_peak_morning": is_peak_morning,
        "is_peak_afternoon": is_peak_afternoon,
        "is_busy_day": is_busy_day,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict care delay for a patient",
    response_description="The predicted care delay in minutes",
)
async def predict_care_delay(patient: PatientData):
    """
    Predict care delay for a patient based on ward conditions and patient attributes.
    
    The input must include patient information and a registration timestamp. Time-related
    features will be automatically extracted from the timestamp.
    
    ## Example
    ```bash
    curl -X 'POST' \\
      'http://localhost:8000/predict' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "age": 65,
        "gender": "M",
        "num_comorbidities": 2,
        "severity_score": 3,
        "primary_diagnosis": "Pneumonia",
        "registration_timestamp": "2025-05-15 10:30:00",
        "total_beds": 30,
        "occupied_beds": 25,
        "staff_count": 5,
        "staff_to_patient_ratio": 5.0
    }'
    ```
    """
    try:
        # Create a dictionary for input data
        data = patient.dict()

        # Extract time features from timestamp
        time_features = extract_time_features(data["registration_timestamp"])

        # Add time features to data
        data.update(time_features)

        # Remove the original timestamp field as it's not needed for the model
        data.pop("registration_timestamp")

        # Calculate derived features if not provided
        if data.get("occupancy_rate") is None:
            data["occupancy_rate"] = (data["occupied_beds"] / data["total_beds"]) * 100

        if data.get("patients_per_staff") is None:
            data["patients_per_staff"] = data["occupied_beds"] / data["staff_count"]

        if data.get("comorbidity_severity") is None:
            data["comorbidity_severity"] = (
                data["num_comorbidities"] * data["severity_score"]
            )

        # Add beds_per_staff which wasn't in the Pydantic model but used by the model
        data["beds_per_staff"] = data["total_beds"] / data["staff_count"]

        # Add high_occupancy indicator
        data["high_occupancy"] = 1 if data["occupancy_rate"] > 85 else 0

        # Create a DataFrame for prediction
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Create simplified response with renamed field
        return {"predicted_care_delay": round(float(prediction), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    description="Returns the health status of the API and model",
)
async def health_check():
    """
    Check the health status of the API and model

    ## Example
    ```bash
    curl -X 'GET' 'http://localhost:8000/health'
    ```
    """
    return {"status": "healthy", "model_loaded": model is not None}


# Entry point for running the application
if __name__ == "__main__":
    import uvicorn

    print("Care Delay Prediction API (FastAPI) starting...")
    print(f"Loaded model: {model_info['model_name']}")
    print(
        f"Model performance - RMSE: {model_info['performance']['rmse']:.2f}, R²: {model_info['performance']['r2']:.3f}"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
