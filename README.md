# NYC Taxi Trip Duration Prediction

## Project Overview

This project aims to predict the duration of taxi trips in New York City using machine learning regression techniques.

The objective is to develop an end-to-end machine learning solution using historical taxi trip data, perform exploratory data analysis, engineer meaningful features, compare regression algorithms, evaluate model performance, and deploy the final model through a REST API.

## Business Problem

Accurate taxi trip duration estimates can help improve trip planning, passenger experience, and transportation operations.

This project explores how historical trip information, including pickup and drop-off locations, timestamps, and passenger details, can be used to estimate journey duration.

## Dataset

Source: [NYC Taxi Trip Duration](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration) — Kaggle

- Original dataset size: Approximately 1.46 million records.

- Target variable: trip_duration

- Target measurement: Seconds.

The dataset contains pickup and drop-off timestamps, geographic coordinates, passenger counts, vendor information, and trip duration.

The raw dataset is not included in this repository.

To run the notebook or training script, download the dataset from Kaggle and place NYC.csv inside the data/ directory.

## Project Workflow

The project follows an end-to-end machine learning regression workflow:

1. Dataset loading and initial inspection.

2. Data quality assessment and cleaning.

3. Exploratory data analysis (EDA).

4. Outlier investigation and treatment.

5. Geographic and temporal feature engineering.

6. Manhattan distance, bearing, and route-related feature engineering.

7. Geographic clustering using K-Means.

8. Chronological train-validation-test splitting.

9. Feature preprocessing and transformation.

10. Regression model training and comparison.

11. Hyperparameter tuning.

12. Model evaluation using MAE, RMSE, and R².

13. Residual analysis and prediction error investigation.

14. Cross-validation and feature importance analysis.

15. Final model selection and training.

16. Saving trained models and evaluation figures.

17. Building a reusable prediction module.

18. Developing a REST API using FastAPI.

19. Containerizing the application using Docker.

20. Deploying the prediction API on Render.

Multiple regression algorithms were evaluated. Training code for some computationally expensive experiments was removed after evaluation, while their recorded results were retained for comparison.

## Models and Techniques

The project explores multiple regression approaches, including:

- Linear Regression

- Ridge Regression

- Lasso Regression

- ElasticNet

- Random Forest Regressor

- Gradient Boosting Regressor

- HistGradientBoosting Regressor

- XGBoost Regressor

Additional techniques include:

- ColumnTransformer-based preprocessing

- K-Means geographic clustering

- Hyperparameter tuning

- Chronological data splitting

- Cross-validation

- Feature importance analysis

## Evaluation

Model performance is evaluated using:

- Mean Absolute Error (MAE)

- Root Mean Squared Error (RMSE)

- R² Score

Additional analysis includes actual-versus-predicted values, residual distribution, and prediction errors across different trip-duration ranges.

Evaluation figures are available in reports/figures/.

## Final Model

The final prediction pipeline uses an XGBoost Regressor with a preprocessing pipeline.

The model incorporates geographic, temporal, and trip-related features, along with pickup and drop-off geographic clusters generated using K-Means.

The trained artifacts are saved in the models/ directory:

- final_model.joblib

- pickup_kmeans.joblib

- dropoff_kmeans.joblib

## Model Performance

The final XGBoost Regressor was evaluated on the unseen test dataset.

| Evaluation Metric | Test Result |
|---|---:|
| **MAE** | 199.481 seconds |
| **RMSE** | 372.043 seconds |
| **R² Score** | 0.746 |

The model achieved an R² score of 0.746, explaining approximately 74.6% of the variance in trip duration on the test dataset.

The MAE indicates an average absolute prediction error of approximately 199.5 seconds (3.32 minutes).

##Repository Structure

```
NYC-taxi-trip-duration/
│
├── NYC_Trip_duration.ipynb
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
│
├── data/
│   └── NYC.csv                  # Local dataset; excluded from Git
│
├── models/
│   ├── final_model.joblib
│   ├── pickup_kmeans.joblib
│   └── dropoff_kmeans.joblib
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── api.py
│
├── tests/
│   └── test_predict.py
│
└── reports/
    └── figures/
        ├── error_by_trip_duration.png
        ├── final_model_actual_vs_predicted_2hours.png
        ├── final_model_actual_vs_predicted.png
        ├── residual_distribution.png
        ├── xgb_feature_importance.png
        ├── docker_api_test.png
        └── render_api_test.png
```
## Setup and Execution

1. Clone the Repository
```
git clone https://github.com/niks813/nyc-taxi-trip-duration.git
cd NYC-Trip-Duration
```
2. Create a Virtual Environment
```
uv venv --python 3.14.7
```
3. Install Dependencies
```
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```
4. Download the Dataset

Download NYC.csv from the Kaggle dataset linked above and place it inside the data/ directory.

5. Run the Notebook

Open NYC_Trip_duration.ipynb in VS Code, select the project's virtual environment as the notebook kernel, and execute the cells in order.

## Model Training

The training workflow is also available as a standalone Python script.

From the project root, execute:
```
.venv\Scripts\python.exe src/train.py
```
This script performs the training workflow and saves the trained model artifacts inside the models/ directory.

## Prediction API

The project includes a FastAPI application that serves predictions using the saved model artifacts.

The API accepts trip information, performs the required feature engineering, and returns the predicted trip duration in seconds and minutes.

## API Endpoint

**Method**: POST

**Endpoint**: /predict

## Sample Request
```
{
  "passenger_count": 1,
  "pickup_longitude": -73.985428,
  "pickup_latitude": 40.748817,
  "dropoff_longitude": -73.985130,
  "dropoff_latitude": 40.758896,
  "pickup_datetime": "2021-03-15 08:30:00",
  "vendor_id": 1,
  "store_and_fwd_flag": "N"
}
```
## Sample Response
```
{
  "predicted_duration_seconds": 466.91,
  "predicted_duration_minutes": 7.78
}
```
## Run the API Locally

From the project root, execute:
```
.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```
The API will be available at:
```
http://127.0.0.1:8000
```
Interactive API documentation:
```
http://127.0.0.1:8000/docs
```
## Docker

The application is containerized using Docker.

Build the Docker Image
```
docker build -t nyc-taxi-prediction .
```
Run the Container
```
docker run -d -p 8000:8000 --name nyc-taxi-container nyc-taxi-prediction
```
Access the API documentation at:
```
http://127.0.0.1:8000/docs
```
Deployment

The prediction API is deployed on Render.

- **API URL**: https://nyc-taxi-trip-duration-y3mc.onrender.com

- **API Documentation**: https://nyc-taxi-trip-duration-y3mc.onrender.com/docs

- **Deployment Platform**: Render

- **Containerization**: Docker

- **API Framework**: FastAPI

The deployed API was tested through its public Swagger documentation, and the prediction endpoint successfully returned HTTP 200 with the expected prediction response.

**Note**: The Render free instance may spin down after inactivity. The first request after a period of inactivity may take longer while the service starts again.

## Tools and Technologies

- Python

- Pandas and NumPy

- Matplotlib and Seaborn

- Scikit-learn

- XGBoost

- Joblib

- FastAPI

- Uvicorn

- Docker

- Jupyter Notebook

- VS Code

- Git and GitHub

- uv

- Render

## Project Results

The project implements an end-to-end machine learning regression solution, from data cleaning and feature engineering to model comparison, evaluation, and final model selection.

The final trained model is integrated into a reusable prediction module and served through a FastAPI REST API.

The application is containerized using Docker and deployed on Render, making the prediction service publicly accessible.

Detailed experiments, implementation, and recorded evaluation results are documented in the Jupyter notebook.

## Key Learnings

- Working with a large, real-world dataset.

- Identifying and handling data quality issues and outliers.

- Engineering geographic and temporal features.

- Applying clustering techniques to geographic data.

- Comparing multiple regression algorithms.

- Performing hyperparameter tuning and cross-validation.

- Evaluating models using regression metrics and residual analysis.

- Building reusable machine learning prediction pipelines.

- Developing REST APIs using FastAPI.

- Containerizing machine learning applications using Docker.

- Deploying machine learning services to a cloud platform.