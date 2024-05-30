import mlflow.pyfunc
import os
import streamlit as st

from src.paths import PARENT_DIR

try:
    # WEATHER_API_KEY = os.environ['VisualCrossing_API_KEY']
    WEATHER_API_KEY = st.secrets['WEATHER_API_KEY']
except FileNotFoundError:
    WEATHER_API_KEY = os.getenv['WEATHER_API_KEY']
except:
    raise Exception(
        'You need to create a .env file in the project root with the WEATHER_API_KEY')


HOPSWORKS_PROJECT_NAME = 'taxi_ride_forecast'

try:
    # HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
    HOPSWORKS_API_KEY = st.secrets['HOPSWORKS_API_KEY']
except FileNotFoundError:
    HOPSWORKS_API_KEY = os.getenv['HOPSWORKS_API_KEY']
except:
    raise Exception(
        'You need to create a .env file in the project root with the HOPSWORKS_API_KEY')


# For Rides
RIDE_FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
RIDE_FEATURE_GROUP_VERSION = 1
RIDE_FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
RIDE_FEATURE_VIEW_VERSION = 1


# For Weather
WEATHER_FEATURE_GROUP_NAME = 'daily_weather_feature_group'
WEATHER_FEATURE_GROUP_VERSION = 1
WEATHER_FEATURE_VIEW_NAME = 'daily_weather__feature_view'
WEATHER_FEATURE_VIEW_VERSION = 1


MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 1


# Added for monitoring purposes
MODEL_PREDICTIONS_FEATURE_GROUP = 'model_predictions_feature_group'
MODEL_PREDICTIONS_FEATURE_GROUP_VERSION = 1
MODEL_PREDICTIONS_FEATURE_VIEW = 'model_predictions_feature_view'
MODEL_PREDICTIONS_FEATURE_VIEW_VERSION = 1
MODEL_FEATURE_VIEW_MONITORING = 'predictions_vs_actuals_for_monitoring_feature_view'

# Number of historical ride values our model needs to generate predictions
N_FEATURES_RIDE = 24 * 3

# Number of historical weather values our model needs to generate predictions
N_FEATURES_WEATHER = 12

# Maximum Mean Absolute Error we allow our production model to have
MAX_MAE = 4.0


# MLflow Model Registry 

MLFLOW_MODEL_NAME = "Taxi-Demand-Predictor"
MLFLOW_MODEL_VERSION = 8
