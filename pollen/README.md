# Pollen Allergy Guardian

## 1. Dynamic Data Sources
This project uses real-time, dynamic data sources:

* **Pollenrapporten API** - Live grass pollen measurements from monitoring stations in Stockholm
* **Open-Meteo API** - Real-time weather data and 7-day forecasts including temperature, precipitation, wind speed, and wind direction

Both APIs provide continuously updated data, ensuring our predictions are based on the most current information available.

## 2. Prediction Problem
**Problem Statement:** Predict grass pollen concentration levels in Stockholm for the next 7 days.

**Why it matters:** Grass pollen allergies affect millions of people during pollen season (May-August). Accurate short-term forecasts help individuals:
- Plan outdoor activities when pollen levels are low
- Take preventive medications before high pollen days
- Adjust daily routines to minimize allergy symptoms

**Technical Approach:** Regression problem using XGBoost to predict pollen levels (0-6 scale) based on weather conditions and engineered features.

## 3. User Interface (UI)
We provide an interactive **Streamlit web application** that delivers prediction value through:

* **7-Day Pollen Forecast Dashboard** - Visual charts showing predicted pollen levels
* **AI-Powered Chat Interface** - Users can ask natural language questions about pollen forecasts
* **LLM Integration** - OpenAI and Hermes models provide personalized advice and explanations
* **Real-time Updates** - Dashboard refreshes with latest predictions from the ML pipeline

**🌐 Live Demo:** [https://mlfs-book-jemcnrpdtmfaaet8idit6r.streamlit.app/](https://mlfs-book-jemcnrpdtmfaaet8idit6r.streamlit.app/)

**Screenshot of the UI:**
<img width="1920" height="919" alt="Streamlit Pollen App" src="https://github.com/user-attachments/assets/cbe8e1e1-f6de-4dc4-a313-5fa9cbbf19df" />

## 4. Technology Overview
**Core Technologies:**

* **Machine Learning:** XGBoost for regression, Scikit-learn for preprocessing
* **Feature Store & Model Registry:** Hopsworks for data management and model versioning
* **Data Pipeline:** Python notebooks for ETL, feature engineering, and batch inference
* **Web Framework:** Streamlit for interactive dashboard
* **AI Integration:** OpenAI API and Hermes LLM for natural language interactions
* **Data Sources:** REST APIs (Pollenrapporten, Open-Meteo)
* **Deployment:** GitHub Actions for automated daily pipelines and CI/CD

**Architecture:**
- **Feature Pipeline:** Daily data ingestion and processing (automated via GitHub Actions)
- **Training Pipeline:** Model training and registration
- **Inference Pipeline:** Batch prediction generation (scheduled daily)
- **UI Layer:** Streamlit app with LLM integration
- **CI/CD:** GitHub Actions for automated workflow execution

## Model Performance
The XGBoost model achieved strong performance on the test set:
* **R² score:** 0.8837 (explains 88% of variance)
* **MAE:** 0.2493 (average error of 0.25 on 0-6 scale)

**Feature Importance Analysis:**
<img width="764" height="455" alt="Feature Importance" src="https://github.com/user-attachments/assets/d83f720a-c435-4e7b-b482-1becc6cfd3eb" />

## Repository Structure
* `1_grass_pollen_feature_backfill.ipynb` - Training pipeline with feature engineering
* `2_grass_pollen_feature_pipeline.ipynb` - Daily data ingestion pipeline
* `3_grass_pollen_batch_inference.ipynb` - Prediction generation pipeline
* `grass_pollen_app_streamlit.py` - Interactive web application
* `functions/` - Helper modules for LLM integration and data retrieval
* `requirements.txt` - Python dependencies
* `.github/workflows/pollen-prediction-daily.yml` - GitHub Actions workflow for automated daily pipeline execution

## Quick Start
**🚀 Try the Live App:** [https://mlfs-book-jemcnrpdtmfaaet8idit6r.streamlit.app/](https://mlfs-book-jemcnrpdtmfaaet8idit6r.streamlit.app/)

**Local Development:**
1. Install dependencies: `pip install -r requirements.txt`
2. Configure Hopsworks credentials in `.env` file
3. Run training pipeline: `1_grass_pollen_feature_backfill.ipynb`
4. Launch web app: `streamlit run grass_pollen_app_streamlit.py`
