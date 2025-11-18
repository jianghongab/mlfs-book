## Predict Air Quality

This project builds an Air Quality Forecasting Service for an Air Quality sensor available at https://waqi.info/.


The output is a forecast for air quality, like this one:

![Air quality Prediction](/docs/air-quality/assets/img/pm25_forecast.png)

![Pm25 Prediction](/notebooks/airquality/air_quality_model/images/pm25_hindcast.png)



## Update Model Performance and Analysis

![Pm25 Prediction](/notebooks/airquality/air_quality_model/images/feature_importance_with_lags.png)

![Pm25 Prediction](/notebooks/airquality/air_quality_model/images/performance_comparison.png)

The addition of lagged air quality features IMPROVED the model performance.
  - MSE decreased by 42.62%, indicating better prediction accuracy
  - R² increased by 23176.29%

Why lagged features help:
  1. Air quality has temporal dependencies - today's pollution affects tomorrow's
  2. Weather patterns and pollution sources have persistence over multiple days
  3. Lagged features capture the 'memory' of the atmospheric system
  4. Previous day's PM2.5 levels are strong predictors of current levels

Model Complexity:
  - Baseline model: 4 features (weather only)
  - Lagged model: 7 features
  - Added 3 lagged features


## Tutorial Instructions

You can find [instructions for running this tutorial in this Google Doc](https://docs.google.com/document/d/1YXfM1_rpo1-jM-lYyb1HpbV9EJPN6i1u6h2rhdPduNE/edit?usp=sharing).

