import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins and all ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Training and saving models (run once)
print("=== Training Flood Risk Prediction Model ===")
df_flood = pd.read_csv("large_flood_data.csv")
high_discharge_threshold = df_flood["river_discharge"].quantile(0.75)
high_precipitation_threshold = df_flood["precipitation"].quantile(0.75)
df_flood["flood_risk"] = ((df_flood["river_discharge"] > high_discharge_threshold) |
                          (df_flood["precipitation"] > high_precipitation_threshold)).astype(int)
X_flood = df_flood[["river_discharge", "precipitation"]]
y_flood = df_flood["flood_risk"]
X_train_flood, X_test_flood, y_train_flood, y_test_flood = train_test_split(X_flood, y_flood, test_size=0.2, random_state=42)
flood_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
flood_model.fit(X_train_flood, y_train_flood)
y_pred_flood = flood_model.predict(X_test_flood)
print(f"Flood Model Accuracy: {accuracy_score(y_test_flood, y_pred_flood) * 100:.2f}%")
flood_model_file = "flood_model.pkl"
joblib.dump(flood_model, flood_model_file)
print(f"Flood model saved as {flood_model_file}")

print("\n=== Training Cyclone Severity Prediction Model ===")
df_cyclone = pd.read_csv("small_cyclone_data.csv")
df_cyclone["wind_speed"] = pd.to_numeric(df_cyclone["wind_speed"], errors='coerce')
df_cyclone["central_pressure"] = pd.to_numeric(df_cyclone["central_pressure"], errors='coerce')
df_cyclone.dropna(subset=["wind_speed", "central_pressure", "latitude", "longitude"], inplace=True)
severe_wind_threshold = 64
df_cyclone["cyclone_severity"] = (df_cyclone["wind_speed"] >= severe_wind_threshold).astype(int)
X_cyclone = df_cyclone[["central_pressure", "latitude", "longitude"]]
y_cyclone = df_cyclone["cyclone_severity"]
X_train_cyclone, X_test_cyclone, y_train_cyclone, y_test_cyclone = train_test_split(X_cyclone, y_cyclone, test_size=0.2, random_state=42)
cyclone_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cyclone_model.fit(X_train_cyclone, y_train_cyclone)
y_pred_cyclone = cyclone_model.predict(X_test_cyclone)
print(f"Cyclone Model Accuracy: {accuracy_score(y_test_cyclone, y_pred_cyclone) * 100:.2f}%")
cyclone_model_file = "cyclone_model.pkl"
joblib.dump(cyclone_model, cyclone_model_file)
print(f"Cyclone model saved as {cyclone_model_file}")

# FastAPI server
if os.path.exists(flood_model_file):
    flood_model = joblib.load(flood_model_file)
    print("Loaded flood model from file")
else:
    raise FileNotFoundError(f"{flood_model_file} not found!")

if os.path.exists(cyclone_model_file):
    cyclone_model = joblib.load(cyclone_model_file)
    print("Loaded cyclone model from file")
else:
    raise FileNotFoundError(f"{cyclone_model_file} not found!")

@app.get("/predict/flood")
async def predict_flood(river_discharge: float, precipitation: float, lat: float, lon: float):
    data = np.array([[river_discharge, precipitation]])
    prediction = flood_model.predict_proba(data)[0][1] * 100
    return {
        "event": "Flood Risk",
        "probability": round(prediction, 2),
        "expectedDate": "Within 24-48 hours",
        "intensity": "High" if prediction > 50 else "Low",
        "image": "https://source.unsplash.com/800x600/?flood,disaster",
        "details": f"Predicted flood risk based on river discharge ({river_discharge}) and precipitation ({precipitation}).",
        "recommendations": ["Prepare sandbags", "Avoid low-lying areas", "Monitor weather updates"],
        "location": f"Lat: {lat}, Lon: {lon}"
    }

@app.get("/predict/cyclone")
async def predict_cyclone(central_pressure: float, lat: float, lon: float):
    data = np.array([[central_pressure, lat, lon]])
    prediction = cyclone_model.predict_proba(data)[0][1] * 100
    return {
        "event": "Cyclone Severity",
        "probability": round(prediction, 2),
        "expectedDate": "Within 24-48 hours",
        "intensity": "Severe" if prediction > 50 else "Not Severe",
        "image": "https://source.unsplash.com/800x600/?cyclone,storm",
        "details": f"Predicted cyclone severity based on central pressure ({central_pressure}) at coordinates ({lat}, {lon}).",
        "recommendations": ["Stay indoors", "Stock supplies", "Follow evacuation orders"],
        "location": f"Lat: {lat}, Lon: {lon}"
    }
