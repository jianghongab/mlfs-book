# Pollen Allergy Guardian

## 🌐 Live Demo
[Try the Streamlit App](https://mlfs-book-jemcnrpdtmfaaet8idit6r.streamlit.app/)

## Project Overview
This project predicts grass pollen levels in Stockholm for the next 7 days using real-time weather data and machine learning.

### Key Features
- **Real-time Data**: Pollenrapporten API + Open-Meteo API
- **ML Model**: XGBoost regression (R² = 0.88)
- **Interactive UI**: Streamlit dashboard with AI chat
- **Automation**: GitHub Actions daily pipeline

### Repository Structure
- `pollen/1_grass_pollen_feature_backfill.ipynb` - Training pipeline
- `pollen/2_grass_pollen_feature_pipeline.ipynb` - Daily data pipeline  
- `pollen/3_grass_pollen_batch_inference.ipynb` - Prediction pipeline
- `pollen/grass_pollen_app_streamlit.py` - Web application

[View Full Documentation](./pollen)