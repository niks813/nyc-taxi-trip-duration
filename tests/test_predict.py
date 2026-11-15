from src.predict import predict_trip_duration

result = predict_trip_duration(
    passenger_count=1,
    pickup_longitude=-73.985428,
    pickup_latitude=40.748817,
    dropoff_longitude=-73.985130,
    dropoff_latitude=40.758896,
    pickup_datetime="2021-03-15 08:30:00",
    vendor_id=1,
    store_and_fwd_flag="N"
)

print("Prediction successful!")
print("Predicted duration:", result)