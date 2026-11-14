# NYC Taxi Trip Duration Prediction

## Project Overview

This project aims to predict the duration of taxi trips in New York City using machine learning regression techniques.

The objective is to develop a predictive model using historical taxi trip data, perform exploratory data analysis, engineer meaningful features, compare regression algorithms, and evaluate model performance.

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

To run the notebook, download the dataset from Kaggle and place NYC.csv inside the data/ directory.

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

## Repository Structure
```
NYC-Trip-duration/
│
├── NYC_Trip_duration.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── NYC.csv                 # Local dataset; excluded from Git
│
├── models/
│   ├── final_model.joblib
│   ├── pickup_kmeans.joblib
│   └── dropoff_kmeans.joblib
│
└── reports/
    └── figures/
        ├── error_by_trip_duration.png
        ├── final_model_actual_vs_predicted_2hours.png
        ├── final_model_actual_vs_predicted.png
        ├── residual_distribution.png
        └── xgb_feature_importance.png
```


## Setup and Execution

### 1. Clone the repository
git clone <repository-url>
cd NYC-taxi-trip-duration


### 2. Create a virtual environment
uv venv --python 3.14.7


### 3. Install dependencies
uv pip install --python .venv/Scripts/python.exe -r requirements.txt


### 4. Download the dataset

Download NYC.csv from the Kaggle dataset linked above and place it inside the data/ directory.

### 5. Run the notebook

Open NYC_Trip_duration.ipynb in VS Code, select the project's virtual environment as the notebook kernel, and execute the cells in order.

## Tools and Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook
- VS Code
- Git and GitHub
- uv
## Project Results

The project implements an end-to-end machine learning regression workflow, from data cleaning and feature engineering to model comparison, evaluation, and final model selection.

The trained model artifacts are saved in the models/ directory, and the evaluation visualizations are organized in reports/figures/.

Detailed experiments, implementation, and recorded evaluation results are documented in the Jupyter notebook.

## Key Learnings
- Working with a large, real-world dataset.
- Identifying and handling data quality issues and outliers.
- Engineering geographic and temporal features.
- Applying clustering techniques to geographic data.
- Comparing multiple regression algorithms.
- Performing hyperparameter tuning and cross-validation.
- Evaluating models using regression metrics and residual analysis.
- Organizing reusable model artifacts and project outputs.