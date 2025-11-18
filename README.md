# Air Quality Prediction System

# Project Overview
This project aims to build an automated Machine Learning pipeline to predict air quality (PM2.5) for a specific city using weather data. The project implements a workflow, covering the entire lifecycle from data ingestion, feature engineering, and model training to model registration, batch inference, and performance monitoring.

For this study, the air quality monitoring station we selected via AQICN is located at Österväg 17 in Visby, Sweden. The corresponding weather data was collected from Open-Meteo.

# Workflow（Key components)

Feature Pipeline
- Backfill: Ingested and cleaned historical weather data (from Open-Meteo API) and air quality data (from AQICN API) to initialize Feature Groups.

- Daily Update: Scheduled via GitHub Actions to run daily. It fetches the previous day's actual measurements and the weather forecast for the next 7-10 days to update the Feature Store in real-time.

Training Pipeline
- Feature Selection: Utilized Hopsworks Feature Views to create point-in-time correct training datasets.

- Model Building: Trained a predictive model using the XGBoost Regressor algorithm.

- Model Registry: Versioned and registered the trained model along with its performance metrics (MSE, R²) to the Hopsworks Model Registry.

Inference & Monitoring Pipeline:

- Batch Inference: Automatically pulls the latest model and weather forecast daily to predict PM2.5 levels for the next 7 days.

- Monitoring (Hindcast): Generates a "Predicted vs. Actual" comparison chart (Hindcast Graph) to continuously monitor the model's performance against real-world data.

The workflow is automatically run daily in GitHub Actions, and the prediction and hindcast graphs are shown on the dashboard, meeting the requirements of Tasks 1-5.


# Lagged Features Implementation (Task 6)

To enhance the model's predictive performance, we implemented lagged features (PM2.5 values from the past 1, 2, and 3 days) in the model. This experiment was conducted in lagged_features_analysis_fixed.ipynb. In this notebook, we trained both a baseline model (using only weather features) and the updated model (with lagged features) for comparison. As shown in the results below, there is a significant improvement, characterized by a lower Mean Squared Error (MSE) and a higher R-squared ($R^2$) score. 

<img width="946" height="368" alt="image" src="https://github.com/user-attachments/assets/e5ee4d69-70c5-4629-b989-c00479e6d5bd" />

Additionally, we analyzed the importance of different features, as shown in the results below.

<img width="1128" height="548" alt="image" src="https://github.com/user-attachments/assets/03221c3a-0a18-4678-8e34-58497cc81cfe" />


The graphs presented above confirm the strong temporal dependency in the data. In time-series forecasting, the value from the previous day is often the single most powerful predictor. These lagged features capture this historical context, enabling the model to understand the persistence of pollution patterns that weather features alone cannot explain.



