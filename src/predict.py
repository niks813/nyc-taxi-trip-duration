from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


# Load trained model and KMeans artifacts
final_model = joblib.load(MODELS_DIR / "final_model.joblib")
pickup_kmeans = joblib.load(MODELS_DIR / "pickup_kmeans.joblib")
dropoff_kmeans = joblib.load(MODELS_DIR / "dropoff_kmeans.joblib")


def predict_trip_duration(
    passenger_count,
    pickup_longitude,
    pickup_latitude,
    dropoff_longitude,
    dropoff_latitude,
    pickup_datetime,
    vendor_id,
    store_and_fwd_flag
):

    # Convert pickup datetime
    pickup_datetime = pd.to_datetime(pickup_datetime)

    # Extract calendar features
    pickup_hour = pickup_datetime.hour
    pickup_day = pickup_datetime.day
    pickup_dayofweek = pickup_datetime.dayofweek
    pickup_month = pickup_datetime.month

    # Identify weekend and rush hour
    is_weekend = int(pickup_dayofweek >= 5)

    is_rush_hour = int(
        pickup_dayofweek < 5
        and (
            7 <= pickup_hour <= 9
            or 16 <= pickup_hour <= 19
        )
    )

    # Calculate Haversine distance
    lat1 = np.radians(pickup_latitude)
    lon1 = np.radians(pickup_longitude)
    lat2 = np.radians(dropoff_latitude)
    lon2 = np.radians(dropoff_longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    distance_km = 6371 * c

    # Calculate Manhattan distance
    lat_diff = np.radians(dropoff_latitude - pickup_latitude)
    lon_diff = np.radians(dropoff_longitude - pickup_longitude)

    mean_lat = np.radians(
        (dropoff_latitude + pickup_latitude) / 2
    )

    lat_distance_km = 6371 * abs(lat_diff)
    lon_distance_km = 6371 * abs(lon_diff) * np.cos(mean_lat)

    manhattan_distance_km = lat_distance_km + lon_distance_km

    # Calculate bearing
    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360

    bearing_sin = np.sin(np.radians(bearing))
    bearing_cos = np.cos(np.radians(bearing))

    # Apply cyclical encoding
    hour_sin = np.sin(2 * np.pi * pickup_hour / 24)
    hour_cos = np.cos(2 * np.pi * pickup_hour / 24)

    dayofweek_sin = np.sin(2 * np.pi * pickup_dayofweek / 7)
    dayofweek_cos = np.cos(2 * np.pi * pickup_dayofweek / 7)

    # Calculate route detour ratio
    route_detour_ratio = (
        manhattan_distance_km / (distance_km + 1e-6)
        if distance_km > 0
        else 0
    )

    # Predict pickup and dropoff zones
    pickup_zone = pickup_kmeans.predict(
    pd.DataFrame(
        [[pickup_latitude, pickup_longitude]],
        columns=['pickup_latitude', 'pickup_longitude']
    )
    )[0]

    dropoff_zone = dropoff_kmeans.predict(
    pd.DataFrame(
        [[dropoff_latitude, dropoff_longitude]],
        columns=['dropoff_latitude', 'dropoff_longitude']
    )
    )[0]

    # Prepare model input
    input_data = pd.DataFrame([{
        'passenger_count': passenger_count,
        'pickup_longitude': pickup_longitude,
        'pickup_latitude': pickup_latitude,
        'dropoff_longitude': dropoff_longitude,
        'dropoff_latitude': dropoff_latitude,
        'distance_km': distance_km,
        'pickup_hour': pickup_hour,
        'pickup_day': pickup_day,
        'pickup_dayofweek': pickup_dayofweek,
        'pickup_month': pickup_month,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'manhattan_distance_km': manhattan_distance_km,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dayofweek_sin': dayofweek_sin,
        'dayofweek_cos': dayofweek_cos,
        'route_detour_ratio': route_detour_ratio,
        'bearing_sin': bearing_sin,
        'bearing_cos': bearing_cos,
        'vendor_id': vendor_id,
        'store_and_fwd_flag': store_and_fwd_flag,
        'pickup_zone': pickup_zone,
        'dropoff_zone': dropoff_zone
    }])

    # Generate prediction
    predicted_seconds = float(final_model.predict(input_data)[0])

    predicted_minutes = predicted_seconds / 60

    return {
        "predicted_duration_seconds": round(predicted_seconds, 2),
        "predicted_duration_minutes": round(predicted_minutes, 2)
    }

if __name__ == "__main__":
    print("Prediction module loaded successfully.")
    print("All model artifacts loaded successfully.")