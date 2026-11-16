from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime

from src.predict import predict_trip_duration


# Initialize the FastAPI application
app = FastAPI(
    title="NYC Taxi Trip Duration Prediction API",
    description="Predicts taxi trip duration using a trained XGBoost model.",
    version="1.0.0"
)


# Define the expected input data
class TripInput(BaseModel):
    passenger_count: int = Field(ge=1, le=6)

    pickup_longitude: float = Field(ge=-180, le=180)
    pickup_latitude: float = Field(ge=-90, le=90)

    dropoff_longitude: float = Field(ge=-180, le=180)
    dropoff_latitude: float = Field(ge=-90, le=90)

    pickup_datetime: datetime

    vendor_id: int = Field(ge=1, le=2)

    store_and_fwd_flag: str = Field(pattern="^[YN]$")


# Root endpoint
@app.get("/")
def home():
    return {"message": "NYC Taxi Trip Duration Prediction API is running."}


# Prediction endpoint
@app.post("/predict")
def predict(input_data: TripInput):

    result = predict_trip_duration(
        passenger_count=input_data.passenger_count,
        pickup_longitude=input_data.pickup_longitude,
        pickup_latitude=input_data.pickup_latitude,
        dropoff_longitude=input_data.dropoff_longitude,
        dropoff_latitude=input_data.dropoff_latitude,
        pickup_datetime=input_data.pickup_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        vendor_id=input_data.vendor_id,
        store_and_fwd_flag=input_data.store_and_fwd_flag
    )

    return result