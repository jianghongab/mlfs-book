import pandas as pd
import datetime
import hopsworks
from hsfs.feature import Feature


def get_historical_pollen_for_date(date: str, feature_view, weather_fg, model) -> pd.DataFrame:
    """
    Retrieves historical grass pollen data for a specific date from a feature view.

    Args:
        date (str): The date in the format "YYYY-MM-DD".
        feature_view: The feature view object.
        weather_fg: The weather feature group.
        model: The machine learning model.

    Returns:
        pd.DataFrame: A DataFrame containing the date and the grass pollen level.
    """
    # Convert date string to datetime object for querying
    date_datetime = datetime.datetime.strptime(date, "%Y-%m-%d")

    # Retrieve observation data for the specified period from the Feature View
    features_df, labels_df = feature_view.training_data(
        start_time=date_datetime, end_time=date_datetime + datetime.timedelta(days=1), statistics_config=False
    )

    # Merge features with labels (pollen concentration values)
    batch_data = features_df.copy()
    batch_data["grass_pollen"] = labels_df["grass_pollen"]

    # Format the date for display
    batch_data["date"] = pd.to_datetime(batch_data["datetime_id"]).dt.strftime("%Y-%m-%d")

    return batch_data[["date", "grass_pollen"]].sort_values("date").reset_index(drop=True)


def get_historical_pollen_in_date_range(date_start: str, date_end: str, feature_view, weather_fg, model) -> pd.DataFrame:
    """
    Retrieves actual grass pollen measurements for a historical date range.
    """
    # Read offline data from the feature view's query object
    batch_data = feature_view.query.read()

    # Ensure date column is tz-naive for comparison
    batch_data["date_dt"] = pd.to_datetime(batch_data["datetime_id"]).dt.tz_localize(None)

    start_dt = pd.to_datetime(date_start)
    end_dt = pd.to_datetime(date_end)

    # Filter data within the date range
    df = batch_data[(batch_data["date_dt"] >= start_dt) & (batch_data["date_dt"] <= end_dt)].copy()
    df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")

    return df[["date", "grass_pollen"]].sort_values("date").reset_index(drop=True)


def get_future_pollen_in_date_range(date_start, date_end, feature_view, weather_fg, model):
    start_dt = datetime.datetime.strptime(date_start, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(date_end, "%Y-%m-%d") if date_end else start_dt

    fg_data = weather_fg.read()
    fg_data["date_dt"] = pd.to_datetime(fg_data["datetime_id"]).dt.tz_localize(None)

    df = fg_data[(fg_data["date_dt"] >= start_dt) & (fg_data["date_dt"] <= end_dt)].copy()

    # Must match your 13 features from 1_grass_pollen_feature_backfill.ipynb
    features = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant",
        "day_of_year",
        "month",
        "is_high_season",
        "gdd_daily",
        "gdd_cumsum",
        "precip_lag_1",
        "temp_lag_1",
        "wind_lag_1",
    ]

    df["grass_pollen"] = model.predict(df[features])
    df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")
    return df[["date", "grass_pollen"]].sort_values("date")


# Helper: single date prediction
def get_future_pollen_for_date(date, feature_view, weather_fg, model):
    return get_future_pollen_in_date_range(date, date, feature_view, weather_fg, model)
