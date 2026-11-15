# Import required libraries and machine learning components

from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

# Define project directories and create the models directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "NYC.csv"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# Convert pickup and dropoff timestamps into datetime format

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

# Calculate geographical distance between pickup and dropoff locations

lat1 = np.radians(df['pickup_latitude'])
lon1 = np.radians(df['pickup_longitude'])
lat2 = np.radians(df['dropoff_latitude'])
lon2 = np.radians(df['dropoff_longitude'])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)

c = 2 * np.arcsin(np.sqrt(a))

df['distance_km'] = 6371 * c

# Calculate average trip speed and remove unrealistic high-speed records

df['avg_speed_kmh'] = (df['distance_km'] / (df['trip_duration'] / 3600))

speed_invalid = df['avg_speed_kmh'] > 120

df = df[~speed_invalid].copy()

# Remove trips with unrealistic durations

duration_invalid = df['trip_duration'] >= 86000

df = df[~duration_invalid].copy()

# Extract time-based features from pickup datetime

df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_dayofweek'] = df['pickup_datetime'].dt.day_of_week
df['pickup_month'] = df['pickup_datetime'].dt.month

# Identify weekends and weekday rush-hour trips

df['is_weekend'] = (df['pickup_dayofweek'] >= 5).astype(int)

df['is_rush_hour'] = (
    (df['pickup_dayofweek'] < 5) & (
        df['pickup_hour'].between(7, 9) | df['pickup_hour'].between(16, 19))
).astype(int)

invalid_duration = df['trip_duration'] > 36000

df = df[~invalid_duration].copy()

print("Data cleaning completed.")
print("Dataset shape after filtering:", df.shape)

# Calculate Manhattan distance as an alternative route-distance feature

lat_diff = np.radians(df['dropoff_latitude'] - df['pickup_latitude'])
lon_diff = np.radians(df['dropoff_longitude'] - df['pickup_longitude'])

mean_lat = np.radians((df['dropoff_latitude'] + df['pickup_latitude']) / 2)

lat_distance_km = 6371 * np.abs(lat_diff)

lon_distance_km = (6371 * np.abs(lon_diff) * np.cos(mean_lat))

df['manhattan_distance_km'] = lat_distance_km + lon_distance_km

# Calculate trip direction and convert bearing into cyclic features

lat1 = np.radians(df['pickup_latitude'])
lon1 = np.radians(df['pickup_longitude'])
lat2 = np.radians(df['dropoff_latitude'])
lon2 = np.radians(df['dropoff_longitude'])

dlon = lon2 - lon1

x = np.sin(dlon) * np.cos(lat2)

y = (np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

df["bearing"] = (np.degrees(np.arctan2(x, y)) + 360) % 360

df['bearing_sin'] = np.sin(np.radians(df['bearing']))
df['bearing_cos'] = np.cos(np.radians(df['bearing']))

# Apply cyclical encoding to hour and day-of-week features

df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)

df['dayofweek_sin'] = np.sin(2 * np.pi * df['pickup_dayofweek'] / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['pickup_dayofweek'] / 7)

# Calculate the ratio between Manhattan and straight-line distances

df['route_detour_ratio'] = np.where(
    df['distance_km'] > 0,
    df['manhattan_distance_km'] / (df['distance_km'] + 1e-6),
    0
)

# Remove intermediate columns that are not required for model training

model_df = df.drop(columns=['id', 'dropoff_datetime', 'avg_speed_kmh', 'bearing']).copy()

# Sort trips chronologically to prevent future data from entering earlier splits

model_df = model_df.sort_values('pickup_datetime').reset_index(drop=True)

# Split data chronologically into training, validation, and test sets

n = len(model_df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = model_df.iloc[:train_end].copy()
val_df = model_df.iloc[train_end:val_end].copy()
test_df = model_df.iloc[val_end:].copy()

print("\nChronological split completed")
print("Training set: ", train_df.shape)
print("Validation set: ", val_df.shape)
print("Test set: ", test_df.shape)

# Separate input features and target variable

target = 'trip_duration'

X_train = train_df.drop(columns=[target, 'pickup_datetime'])
y_train = train_df[target]

X_val = val_df.drop(columns=[target, 'pickup_datetime'])
y_val = val_df[target]

X_test = test_df.drop(columns=[target, 'pickup_datetime'])
y_test = test_df[target]

# Combine training and validation data for final model training

X_train_final = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_train_final = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

X_train_final = X_train_final.drop(columns=['pickup_zone', 'dropoff_zone'], errors='ignore')
X_test_final = X_test.drop(columns=['pickup_zone', 'dropoff_zone'], errors='ignore')

# Train pickup and dropoff location clusters using KMeans

pickup_kmeans_final = KMeans(n_clusters=50, random_state=1, n_init=10)
dropoff_kmeans_final = KMeans(n_clusters=50, random_state=1, n_init=10)

print("Training pickup and dropoff KMeans models...")

pickup_kmeans_final.fit(X_train_final[['pickup_latitude', 'pickup_longitude']])

dropoff_kmeans_final.fit(X_train_final[['dropoff_latitude', 'dropoff_longitude']])

# Assign pickup and dropoff cluster labels to training and test data

X_train_final['pickup_zone'] = pickup_kmeans_final.predict(X_train_final[['pickup_latitude', 'pickup_longitude']])

X_train_final['dropoff_zone'] = dropoff_kmeans_final.predict(X_train_final[['dropoff_latitude', 'dropoff_longitude']])

X_test_final['pickup_zone'] = pickup_kmeans_final.predict(X_test_final[['pickup_latitude', 'pickup_longitude']])

X_test_final['dropoff_zone'] = dropoff_kmeans_final.predict(X_test_final[['dropoff_latitude', 'dropoff_longitude']])

print("\nFinal KMeans models trained.")

# Define numerical and categorical features used by the model

numeric_features = [
 'passenger_count',
 'pickup_longitude',
 'pickup_latitude',
 'dropoff_longitude',
 'dropoff_latitude',
 'distance_km',
 'pickup_hour',
 'pickup_day',
 'pickup_dayofweek',
 'pickup_month',
 'is_weekend',
 'is_rush_hour',
 'manhattan_distance_km',
 'hour_sin',
 'hour_cos',
 'dayofweek_sin',
 'dayofweek_cos',
 'route_detour_ratio',
 'bearing_sin',
 'bearing_cos']

categorical_features = [
     'vendor_id',
     'store_and_fwd_flag',
     'pickup_zone',
     'dropoff_zone'
 ]

# Scale numerical features and one-hot encode categorical features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Build the final machine learning pipeline using XGBoost

final_model = Pipeline(
                steps=[
                    ('preprocessor', preprocessor),
                    ('model', XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.1,
                        max_depth=8,
                        min_child_weight=10,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        gamma=0,
                        reg_alpha=0,
                        reg_lambda=1.0,
                        objective='reg:squarederror',
                        random_state=1,
                        n_jobs=-1
                    ))
                ]
              )

# Train the final model using combined training and validation data

print("Starting final XGBoost model training...")

final_model.fit(X_train_final, y_train_final)

print("Final model training completed.")

# Save the trained model and KMeans clustering models for deployment

joblib.dump(final_model, MODELS_DIR / "final_model.joblib")

joblib.dump(pickup_kmeans_final, MODELS_DIR /  "pickup_kmeans.joblib")

joblib.dump(dropoff_kmeans_final, MODELS_DIR / "dropoff_kmeans.joblib")

print("All model artifacts saved successfully.")
print("Saved files:", list(MODELS_DIR.iterdir()))