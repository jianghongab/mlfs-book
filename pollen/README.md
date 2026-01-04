# Pollen Allergy Guardian

## Project Overview
This project focuses on predicting grass pollen levels in Stockholm for the upcoming 7 days using historical pollen data and weather forecasts. Pollen allergies affect a large number of people during the grass pollen season, and short-term forecasts can help individuals better plan daily activities and take preventive measures when pollen levels are expected to be high.

In addition to numerical predictions, a language-model-based user interface is included to make the results easier to understand and interact with.

## System Architecture
The system is implemented in Python and uses Hopsworks as the central Feature Store and Model Registry. The workflow is divided into three pipelines:

* **Feature Pipeline**  
  Runs daily to fetch updated weather data and historical pollen measurements. Raw data is processed into structured features and stored in the online Feature Store.

* **Training Pipeline**  
  Retrieves the full historical dataset from the Feature Store, trains a regression model, evaluates its performance, and registers the trained model in the Model Registry.

* **Batch Inference Pipeline**  
  Loads the latest trained model and weather forecast data to generate pollen level predictions for the next 7 days. The results are written back to the Feature Store for downstream use.

## Data Processing and Feature Engineering
The model is based on two main data sources:
* Historical grass pollen observations  
* Meteorological data, including temperature, precipitation, and wind speed  

Instead of relying only on raw weather variables, we focused on feature engineering to better capture the biological processes behind pollen release.

Key engineered features include:

* **Lag Features**  
  Weather variables from the previous day were added, as pollen levels are often influenced by earlier weather conditions rather than same-day measurements.

* **Growing Degree Days (GDD)**  
  Both daily GDD and cumulative GDD were computed. These features represent accumulated heat exposure and are commonly used to track plant development stages. In our experiments, GDD features provided useful signals for identifying periods when grass flowering is more likely to occur.

## Model and Performance
The task is formulated as a regression problem, and XGBoost was chosen as the prediction model. XGBoost performs well on tabular data and can capture non-linear relationships between weather conditions and pollen levels without excessive complexity.

The dataset was split into training and testing sets to evaluate generalization performance. On the test set, the model achieved:
* **R² score:** 0.8837  
* **MAE (Mean Absolute Error):** 0.2493  

These results indicate that the model explains a large proportion of the variance in pollen levels and provides reasonably accurate short-term forecasts.

To interpret the model's behavior, we analyzed the feature importance (shown below). The plot confirms that the engineered features, such as the cumulative GDD and wind speed lags, contribute significantly to the predictive power of the model.
<img width="764" height="455" alt="575a1bff-d1aa-4155-aa07-383973e28cd3" src="https://github.com/user-attachments/assets/d83f720a-c435-4e7b-b482-1becc6cfd3eb" />



## User Interface and AI Integration
To make the predictions more accessible to users, a web application was developed using Streamlit. Large language models (LLMs), such as OpenAI and Hermes, are integrated to generate short natural-language summaries based on the forecast results.

Users can select the LLM model and ask questions related to pollen levels. The system responds with explanations or warnings derived from the predicted pollen levels. The generated content is intended to provide general guidance rather than medical advice.

The screenshot below demonstrates a sample interaction within the application:
<img width="1920" height="919" alt="5658afa5-1c2d-4892-bbc8-4b334dc9b36f" src="https://github.com/user-attachments/assets/cbe8e1e1-f6de-4dc4-a313-5fa9cbbf19df" />


## Repository Structure
The repository is organized as follows:

* `1_grass_pollen_feature_backfill.ipynb`  
  Initial data processing, feature engineering (Lag features and GDD), model training, and model registration.

* `2_grass_pollen_feature_pipeline.ipynb`  
  Daily feature pipeline that updates the Feature Store with new data.

* `3_grass_pollen_batch_inference.ipynb`  
  Batch inference pipeline that generates 7-day pollen forecasts.

* `grass_pollen_app_streamlit.py`  
  Streamlit application for visualization and user interaction.

* `llm_chain.py` & `context_engineering.py`  
  Modules responsible for formatting inputs and interacting with the LLM.

* `pollen_data_retrieval.py`  
  Helper functions for retrieving prediction results from the Feature Store.
