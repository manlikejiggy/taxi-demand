import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from typing import Optional, Literal

from tqdm import tqdm
import hopsworks
import mlflow
import joblib

from src.paths import MODELS_DIR
import src.config as config
from src.feature_store import get_feature_store


def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )



def load_batch_of_rides_features_from_store(current_date: datetime) -> pd.DataFrame:
    """
    Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_hour`
            - `rides`
            - `pickup_location_id`
    """

    n_features = config.N_FEATURES_RIDE
    feature_store = get_feature_store()

    # Connect to ride feature group
    ride_feature_group = feature_store.get_feature_group(
        name=config.RIDE_FEATURE_GROUP_NAME, 
        version=config.RIDE_FEATURE_GROUP_VERSION
        )
    try:
        # create ride_data feature view if it doesn't exist yet
        feature_store.create_feature_view(
            name=config.RIDE_FEATURE_VIEW_NAME,
            version=config.RIDE_FEATURE_VIEW_VERSION,
            query=ride_feature_group.select_all()
        )
    except:
        print('Ride data feature view already exist. Skiping creation...')

    # Read time-series data from feature store

    fetch_data_from = pd.to_datetime(current_date - timedelta(days=3), utc=True)
    fetch_data_to = pd.to_datetime(current_date - timedelta(hours=1), utc=True)
    
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')

    feature_view = feature_store.get_feature_view(
        name = config.RIDE_FEATURE_VIEW_NAME,
        version = config.RIDE_FEATURE_VIEW_VERSION
        )

    ts_data = feature_view.get_batch_data(
        start_time = pd.to_datetime(fetch_data_from - timedelta(days=1), utc=True),
        end_time = pd.to_datetime(fetch_data_to + timedelta(days=1), utc=True)
        )

    # Convert to UTC aware datetime
    ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], utc=True)

    # Filter data  to the time period we are interested in
    ts_data = ts_data[ts_data['pickup_hour'].between(fetch_data_from, fetch_data_to)]


    # Validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()

    
    print(f'Expected total data points: {n_features*len(location_ids)}')
    print(f'Actual total data points: {len(ts_data)}')

    
    assert len(ts_data) == n_features*len(location_ids), \
        "Time-series data is not complete. Make sure your feature pipeline is up and runnning."


    # Sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

    # Transpose time-series data as a feature vector, for each `pickup_location_id`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)

    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data['pickup_location_id'] == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['ride_demand'].values

    # Numpy arrays to Pandas dataframes
    features = pd.DataFrame(x, columns=[f"rides_previous_{i+1}_hour" for i in reversed(range(n_features))]
                            )

    features['pickup_hour'] = pd.to_datetime(current_date, utc=True)
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features


def load_batch_of_weather_features_from_store(current_date: datetime) -> pd.DataFrame:
    """
    Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame
    """

    feature_store = get_feature_store()

    weather_feature_group = feature_store.get_feature_group(
        name=config.WEATHER_FEATURE_GROUP_NAME, 
        version=config.WEATHER_FEATURE_GROUP_VERSION
        )
    try:
        # create ride_data feature view if it doesn't exist yet
        feature_store.create_feature_view(
            name=config.WEATHER_FEATURE_VIEW_NAME,
            version=config.WEATHER_FEATURE_VIEW_VERSION,
            query=weather_feature_group.select_all()
        )
    except:
        print('Weather data feature view already exist. Skiping creation...')

    # Read time-series data from feature store
    fetch_data_from = pd.to_datetime(current_date - timedelta(days=4), utc=True)
    fetch_data_to = pd.to_datetime(current_date - timedelta(hours=1), utc=True)
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')

    feature_view = feature_store.get_feature_view(
        name=config.WEATHER_FEATURE_VIEW_NAME,
        version=config.WEATHER_FEATURE_VIEW_VERSION
    )

    try:
        weather_data = feature_view.get_batch_data(
            start_time=pd.to_datetime(
                fetch_data_from - timedelta(days=1), utc=True),
            end_time=pd.to_datetime(
                fetch_data_to + timedelta(days=1), utc=True)
        )
    except Exception as e:
        print("Error while fetching data:", e)

    if weather_data.empty:
        print("No data found for the given date range.")

    # Convert to UTC aware datetime
    weather_data['datetime'] = pd.to_datetime(
        weather_data['datetime'], utc=True)

    # Filter data  to the time period we are interested in
    weather_data = weather_data[weather_data['datetime'].between(
        fetch_data_from, fetch_data_to)]

    # Sort data by location and time
    weather_data.sort_values(by=['datetime'], inplace=True)

    return weather_data


def get_spatio_temporal_data(rides: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Combines ride and weather data on dates, returning a merged DataFrame.

    Parameters:
        - rides (pd.DataFrame): Ride data with 'pickup_hour' datetime column.
        - weather (pd.DataFrame): Weather data with 'datetime' datetime column.

    Returns:
        - pd.DataFrame: The combined data set with aligned ride and weather entries.
    """
    rides_df = rides.copy()
    weather_df = weather.copy()

    # convert to date column
    rides_df['pickup_hour'] = rides_df['pickup_hour'].dt.date
    weather_df['datetime'] = weather_df['datetime'].dt.date

    merged_df = pd.merge(rides_df, weather_df, left_on='pickup_hour', right_on='datetime', how='inner')
    merged_df['pickup_hour'] = pd.to_datetime(merged_df['pickup_hour'])
    merged_df.drop('datetime', axis=1, inplace=True)

    return merged_df


def prepare_spatio_temporal_data_for_prediction(rides: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms new ride and weather data for prediction using a serialized pipeline.

    Parameters:
    - rides (pd.DataFrame): New ride data.
    - weather (pd.DataFrame): Corresponding new weather data.

    Returns:
    - pd.DataFrame: Features ready for prediction.
    """

    # Load serialized model artifacts
    artifact = joblib.load(f"{MODELS_DIR}/taxi-demand-artifact.joblib")
    spatio_temporal_data = get_spatio_temporal_data(rides, weather)

    if spatio_temporal_data.empty:
        raise ValueError("The resulting DataFrame from merging rides and weather is empty.")

    pickup_location_id = spatio_temporal_data['pickup_location_id'].values

    data = artifact['pipeline'].transform(spatio_temporal_data)

    col_names = artifact['pipeline'].named_steps['data_preprocessing'].get_feature_names_out()
    dense_array = data.toarray()  # <--- convert from sparse to dense array

    X_test = pd.DataFrame(data=dense_array, columns=col_names)
    X_test['pickup_location_id'] = pickup_location_id

    return X_test


def load_model(source: Literal['local', 'model_registry'] = 'model_registry'):
    """
    Loads a machine learning model for making predictions.

    Parameters:
    - source (Literal['local', 'model_registry']): The source from which to load the model.
      Only 'local' or 'model_registry' are valid. Defaults to 'model_registry'.

    Returns:
    - The loaded model ready for making predictions.
    """
    if source == 'local':
        model = joblib.load(
            f"{MODELS_DIR}/taxi-demand-artifact.joblib")["model"]

    elif source == 'model_registry':
        try:
            model_name = config.MLFLOW_MODEL_NAME
            model_version = config.MLFLOW_MODEL_VERSION
            model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{model_name}/{model_version}")
        except:
            print(
                'Unable to load model from MLflow. Ensure mlflow server is running properly')
    else:
        raise Exception("source must either be 'local' or 'model-registry'")

    return model


def get_model_prediction(model, features: pd.DataFrame) -> pd.DataFrame:
    X_test = features.drop(['pickup_location_id'], axis=1)
    predictions = np.round(np.expm1(model.predict(X_test)))

    result_df = pd.DataFrame()
    result_df['pickup_location_id'] = features['pickup_location_id'].values
    result_df['predicted_demand'] = np.abs(predictions)

    return result_df

def load_rides_prediction_from_store(from_pickup_hour: datetime, to_pickup_hour: datetime) -> pd.DataFrame:
    """
    Connects to the feature store and retrieves model predictions for all
    `pickup_location_id`s and for the time period from `from_pickup_hour`
    to `to_pickup_hour`

    Args:
        from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
    """
    feature_store = get_feature_store()

    predictiong_fg = feature_store.get_feature_group(
        name=config.MODEL_PREDICTIONS_FEATURE_GROUP,
        version=config.MODEL_PREDICTIONS_FEATURE_GROUP_VERSION
    )

    try:
        # Create feature view if it doesn't exist yet
        feature_store.create_feature_view(
            name=config.MODEL_PREDICTIONS_FEATURE_VIEW,
            version=config.MODEL_PREDICTIONS_FEATURE_VIEW_VERSION,
            query=predictiong_fg.select_all()
        )

    except:
        print(f'Feature view {config.MODEL_PREDICTIONS_FEATURE_VIEW} \
              This feature view already exist. Skipping creation...')

    # Get model prediction feature view
    predictions_fv = feature_store.get_feature_view(
        name=config.MODEL_PREDICTIONS_FEATURE_VIEW,
        version=config.MODEL_PREDICTIONS_FEATURE_VIEW_VERSION,
    )

    print(
        f'Fetching predictions for `pickup_hours` between {from_pickup_hour}  and {to_pickup_hour}')

    predictions_df = predictions_fv.get_batch_data(
        start_time=from_pickup_hour - timedelta(days=1),
        end_time=to_pickup_hour + timedelta(days=1))

    # Ensure our datetimes are UTC aware
    predictions_df['pickup_hour'] = pd.to_datetime(
        predictions_df['pickup_hour'], utc=True)

    from_pickup_hour = pd.to_datetime(from_pickup_hour, utc=True)
    to_pickup_hour = pd.to_datetime(to_pickup_hour, utc=True)

    predictions_df = predictions_df[predictions_df['pickup_hour'].between(
        from_pickup_hour, to_pickup_hour)]

    # Sort by `pick_up_hour` and `pickup_location_id`
    predictions_df.sort_values(
        by=['pickup_hour', 'pickup_location_id'], inplace=True)

    return predictions_df