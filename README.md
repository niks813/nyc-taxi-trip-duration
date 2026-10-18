# NYC Taxi Trip Duration Prediction

## Project Overview

This project aims to predict the duration of taxi trips in New York City using machine learning regression techniques.

The goal is to explore trip data, identify meaningful patterns, engineer useful features, and develop predictive models that estimate trip duration from available trip information.

## Business Problem

Accurate trip duration estimates can help improve trip planning, passenger experience, and transportation operations. This project investigates how historical taxi trip information can be used to estimate journey duration.

## Dataset

Source: [NYC Taxi Trip Duration](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration) — Kaggle

Dataset size: Approximately 1.46 million records before cleaning.

Target variable: trip_duration, measured in seconds.

The dataset contains trip timestamps, pickup and drop-off coordinates, passenger counts, vendor information, and trip duration.

The raw dataset is not included in this repository. Download it from the source and place the CSV at data/NYC.csv before running the notebook.

## Project Workflow

- Data loading and initial inspection

- Exploratory data analysis (EDA)

- Data quality checks and cleaning

- Outlier investigation and treatment

- Feature engineering

- Train-validation-test splitting

- Regression model training and comparison

- Hyperparameter tuning

- Model evaluation and error analysis

- Final conclusions and potential improvements

- The workflow will be updated as each stage is completed.

## Tools and Technologies

- Python

- Pandas and NumPy

- Matplotlib and Seaborn

- Scikit-learn

- Jupyter Notebook

- Git and GitHub

## Repository Structure

nyc-taxi-trip-duration/
├── NYC_Trip_duration.ipynb
├── README.md
├── .gitignore
└── data/
    └── NYC.csv  # Local dataset; excluded from Git


## Current Status

Project has been cleaned further and after rerunning it all cells are working fine. Work for writing down markdown instructions before relevant cells have begun which will explain the project and its working in detail. In future cleaning the project and all details with updating readme file and commiting all relevant files to the github will be done.


## Project Results

This project completed an end-to-end machine learning regression workflow for predicting NYC taxi trip durations, including data preprocessing, feature engineering, regression model comparison, and model evaluation.

The notebook contains the implementation, experiments, and evaluation results.

## Key Learnings

- Working with a large, real-world dataset.

- Data quality assessment and outlier treatment.

- Geospatial and datetime feature engineering.

- Comparing regression algorithms using evaluation metrics.

- Evaluating model performance to support model selection.