import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from pdb import set_trace as stop

import shapefile

import numpy as np
import pandas as pd
from io import StringIO
import calendar
import requests
from tqdm import tqdm


from src.paths import PARENT_DIR, RAW_DATA_DIR, TRANSFORMED_DATA_DIR, RAW_WEATHER_DATA_DIR
from src.config import WEATHER_API_KEY 

def remove_duplicate(df: pd.DataFrame, holdout_set: str = ''):
    """
    Detect and remove duplicate rows in the dataset.
    """
    b4 = df.shape[0]
    print(f'{holdout_set} -- Before Removing Duplicate: {b4:,}')
    df.drop_duplicates(keep='first', inplace=True)
    after = df.shape[0]
    print(f'{holdout_set} -- After Removing Duplicate: {after:,}', '\n')

    if b4 == after:
        print(f"There are no duplicate rows in the {holdout_set}", '\n')
    else:
        print(str(b4 - after) + ' ' + "duplicate row(s) has been removed")


def download_one_file_of_raw_data(year: int, month: int) -> Path:
    # API to download NYC TLC dataset
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'

        with open(path, 'wb') as file:
            file.write(response.content)
        return path
    else:
        raise Exception(f'{URL} is not available')


def validate_raw_data(rides_df: pd.DataFrame, month: int, year: int) -> pd.DataFrame:
    """
    Removes rows with pickup_datetimes outside their valid range
    """
    # keep only rides for this month
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides_df = rides_df[rides_df['pickup_datetime'] >= this_month_start]
    rides_df = rides_df[rides_df['pickup_datetime'] < next_month_start]

    return rides_df


def download_one_file_of_raw_weather_data(year: int, month: int) -> Path:
    """
    Downloads weather data for New York City boroughs for a given year and month from 
    the Visual Crossing Weather API and saves it as a Parquet file in a local directory.

    Parameters:
    - year (int): The year for which the data is to be downloaded.
    - month (int): The month for which the data is to be downloaded.

    Returns:
    - Path: The file path to the saved Parquet file.

    The API key used here is a public API key with a daily limit of 1000 requests.
    """

    _, last_day = calendar.monthrange(year, month)
    start_date = f"{year}-{month:02d}-01"
    stop_date = f"{year}-{month:02d}-{last_day:02d}"

    # load key-value pairs from .env file located in the parent directory

    # API to download NYC WEATHER dataset
    URL = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/New%20York%20City%20boroughs/{start_date}/{stop_date}'
    params = {
        'unitGroup': 'metric',
        'include': 'days',
        # Public API Key (limit: 1000 rqst daily)
        'key': WEATHER_API_KEY,
        'contentType': 'csv'}

    response = requests.get(URL, params=params)

    if response.status_code == 200:
        path = RAW_WEATHER_DATA_DIR / f'weather_{year}-{month:02d}.parquet'
        df = pd.read_csv(StringIO(response.text))
        df.to_parquet(path, index=False)

        return path

    else:
        raise Exception(f'{URL} is not available')


def validate_raw_weather_data(weather_df: pd.DataFrame, month: int, year: int) -> pd.DataFrame:
    """
    Removes rows with datetimes outside their valid range
    """
    # keep only weather for this month
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    weather_df = weather_df[weather_df['datetime'] >= this_month_start]
    weather_df = weather_df[weather_df['datetime'] < next_month_start]

    return weather_df


def load_raw_data(year: int, months: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Loads raw data from local storage or downloads it from the NYC website, and
    then loads it into a Pandas DataFrame

    Args:
        year: year of the data to download
        months: months of the data to download. If `None`, download all months

    Returns:
        pd.DataFrame: DataFrame with the following columns:
            - pickup_datetime: datetime of the pickup
            - pickup_location_id: ID of the pickup location
    """
    rides_df = pd.DataFrame()

    if months is None:
        # download data for the entire year (all months)
        months = list(range(1, 13))
    elif isinstance(months, int):
        # download data only for the month specified by the int `month`
        months = [months]
    elif isinstance(months, list):
        # download data only for the month specified by the int `month`
        months = months

    for month in months:
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                # download the file from the NYC website
                print(f'Downloading file {year}-{month:02d}')
                download_one_file_of_raw_data(year, month)
            except:
                print(f'{year}-{month:02d} file is not available')
                continue
        else:
            print(f'{year}-{month:02d} file already exist in local storage')

        # load the file into Pandas
        rides_one_month = pd.read_parquet(local_file)

        # rename columns
        rides_one_month = rides_one_month[[
            'tpep_pickup_datetime', 'PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'PULocationID': 'pickup_location_id'},
            inplace=True
        )

        # validate the file
        rides_one_month = validate_raw_data(
            rides_one_month, month=month, year=year)

        # append to existing data
        rides_df = pd.concat([rides_df, rides_one_month])

    if rides_df.empty:
        # no data, so we return an empty dataframe
        return pd.DataFrame()
    else:
        # keep only time and origin of the ride
        rides_df = rides_df[['pickup_datetime', 'pickup_location_id']]
        # This will be timezone-aware (UTC)
        rides_df['pickup_datetime'] = pd.to_datetime(rides_df['pickup_datetime'], utc=True)
        return rides_df


def load_raw_weather_data(year: int, months: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Loads weather data from local storage or downloads it from VisualCrossing, and
    then loads it into a Pandas DataFrame

    Args:
        year: year of the data to download
        months: months of the data to download. If `None`, download all months

    Returns:
        pd.DataFrame: DataFrame with the following columns:
            - pickup_datetime: datetime of the pickup
            - pickup_location_id: ID of the pickup location
    """
    weather_df = pd.DataFrame()

    if months is None:
        # download data for the entire year (all months)
        months = list(range(1, 13))
    elif isinstance(months, int):
        # download data only for the month specified by the int `month`
        months = [months]
    elif isinstance(months, list):
        # download data only for the month specified by the int `month`
        months = months

    for month in months:
        local_file = RAW_WEATHER_DATA_DIR / \
            f'weather_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                # download the file from the visualcrossing website
                print(f'Downloading file {year}-{month:02d}')
                download_one_file_of_raw_weather_data(year, month)
            except:
                print("Oops!", sys.exc_info()[0], "occurred.")
                print(f'{year}-{month:02d} file is not available')
                continue
        else:
            print(f'{year}-{month:02d} file already exist in local storage')

        # load the file into Pandas
        weather_one_month = pd.read_parquet(local_file)

        # Select columns of interest
        selected_col = ['datetime', 'temp', 'feelslike', 'humidity', 'windspeed', 'dew',
                        'precip', 'snow', 'snowdepth', 'windgust', 'visibility', 'icon'
                        ]
        weather_one_month = weather_one_month[selected_col].copy()

        # validate the file
        weather_one_month = validate_raw_weather_data(
            weather_one_month, month=month, year=year)

        # append to existing data
        weather_df = pd.concat([weather_df, weather_one_month])

    if weather_df.empty:
        # no data, so we return an empty dataframe
        return pd.DataFrame()
    else:
        # This will be timezone-aware (UTC)
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'],  utc=True)  
        return weather_df


def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add necessary rows to the input 'ts_data' to make sure the output
    has a complete list of
    - pickup_hours
    - pickup_location_ids
    """
    location_ids = range(1, ts_data['pickup_location_id'].max() + 1)
    full_range = pd.date_range(
        ts_data['pickup_hour'].min(), ts_data['pickup_hour'].max(), freq='h')

    output = pd.DataFrame()
    for location_id in tqdm(location_ids, colour='green'):

        # keep only rides for this 'location_id'
        ts_data_i = ts_data.loc[ts_data['pickup_location_id']
                                == location_id, ['pickup_hour', 'ride_demand']]

        if ts_data_i.empty:
            # add a dummy entry with a 0
            ts_data_i = pd.DataFrame.from_dict([
                {'pickup_hour': ts_data['pickup_hour'].max(), 'ride_demand': 0}
            ])

        # Quick way to add missing dates with 0 in a Series (ref: https://stackoverflow.com/a/19324591)
        ts_data_i.set_index('pickup_hour', inplace=True)
        ts_data_i.index = pd.DatetimeIndex(ts_data_i.index)
        ts_data_i = ts_data_i.reindex(full_range, fill_value=0)

        # add back `location_id` columns
        ts_data_i['pickup_location_id'] = location_id

        output = pd.concat([output, ts_data_i])

    # move the pickup_hour from the index to a dataframe column
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})

    return output


def transform_raw_data_into_ts_data(rides_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw ride data by location and hour, including time slots with zero rides.

    Parameters:
    - rides_df (pd.DataFrame): Contains 'pickup_datetime' and 'pickup_location_id'.

    Returns:
    - pd.DataFrame: Aggregated data with columns 'pickup_hour', 'pickup_location_id', 'ride_demand'.
    """

    # sum rides per location and pickup_hour
    rides_df['pickup_hour'] = rides_df['pickup_datetime'].dt.floor('h')
    agg_rides = rides_df.groupby(
        ['pickup_hour', 'pickup_location_id']).size().reset_index(name='ride_demand')

    # add rows for (locations, pickup_hours)s with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots


def get_cutoff_indices_features_and_target(data: pd.DataFrame, input_seq_len: int, step_size: int) -> list:

    stop_position = len(data) - 1

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = input_seq_len
    subseq_last_idx = input_seq_len + 1
    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices


def transform_ts_data_into_features_and_target(
        ts_data: pd.DataFrame,
        input_seq_len: int,
        step_size: int) -> pd.DataFrame:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models

    Parameters:
    - ts_data (pd.DataFrame): Contains 'pickup_hour', 'ride_demand', and 'pickup_location_id'.
    - input_seq_len (int): Number of previous hours to generate as features
    - step_size (int): Number of steps to use in generating the rolling window
    """
    assert set(ts_data.columns) == {
        'pickup_hour', 'ride_demand', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for location_id in tqdm(location_ids, colour='green'):

        # keep only ts data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data['pickup_location_id'] == location_id, [
                'pickup_hour', 'ride_demand', ]
        ].sort_values(by=['pickup_hour'])

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location, input_seq_len, step_size)

        # slice and transpose data into numpy arrays for features and targets
        sample_size = len(indices)
        x = np.zeros(shape=(sample_size, input_seq_len), dtype=np.float32)
        y = np.zeros(shape=(sample_size), dtype=np.float32)
        ride_pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['ride_demand'].values
            y[i] = ts_data_one_location.iloc[idx[1]
                : idx[2]]['ride_demand'].values[0]
            ride_pickup_hours.append(
                ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x, columns=[
                f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = ride_pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        targets_one_location = pd.DataFrame(
            y, columns=['target_rides_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])
        
    # convert the pickup_hour from string to datetime
    features['pickup_hour'] = pd.to_datetime(features['pickup_hour'], utc=True)

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']


def fetch_ride_events_from_data_warehouse(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    This function is used to simulate production data by sampling historical data
    from 52 weeks ago (i.e. 1 year)
    """
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)
    print(f'Fetching ride events from {from_date} to {to_date}')

    if (from_date_.year == to_date_.year) and (from_date_.month == to_date_.month):
        # download 1 file of data only
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides['pickup_datetime'] >= from_date_]
        rides = rides[rides['pickup_datetime'] < to_date_]

    else:
        # download 2 files from website
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides['pickup_datetime'] >= from_date_]
        rides_2 = load_raw_data(year=to_date_.year, months=to_date_.month)
        rides_2 = rides_2[rides_2['pickup_datetime'] < to_date_]
        rides = pd.concat([rides, rides_2])

    # shift the pickup_datetime back 1 year ahead, to simulate production data
    # using its 7*52-days-ago value
    rides['pickup_datetime'] += timedelta(days=7*52)

    rides.sort_values(
        by=['pickup_location_id', 'pickup_datetime'], inplace=True)

    return rides


def fetch_batch_raw_ride_data(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data from 52 weeks ago (i.e. 1 year)
    """
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)
    print(f'Fetching ride events from {from_date} to {to_date}')

    # Try to load the data from the first year" 
    rides = load_raw_data(year=from_date_.year)
    rides = rides[(rides['pickup_datetime'] >= from_date_) & (rides['pickup_datetime'] < to_date_)]

    # Check if the years are different"
    if from_date_.year != to_date_.year:
        # Load the data from the second year"
        rides_2 = load_raw_data(year=to_date_.year)
        rides_2 = rides_2[(rides_2['pickup_datetime'] >= from_date_) & (rides_2['pickup_datetime'] < to_date_)]
        rides = pd.concat([rides, rides_2]) 


    # Shift the data to pretend this is recent data
    rides['pickup_datetime'] += timedelta(days=7*52)

    rides.sort_values(by=['pickup_location_id', 'pickup_datetime'], inplace=True)

    return rides





def get_lat_lon(sf: shapefile, shp_dic):
    """
    Get the longitude and logitude of each location
    """
    content = []
    for sr in sf.shapeRecords():
        shape = sr.shape
        rec = sr.record
        loc_id = rec[shp_dic['LocationID']]

        x = (shape.bbox[0]+shape.bbox[2])/2
        y = (shape.bbox[1]+shape.bbox[3])/2

        content.append((loc_id, x, y))
    return pd.DataFrame(content, columns=["LocationID", "longitude", "latitude"])


def add_descriptive_columns(df, basic=True, seasons=True, weekday=True, daytime=True):
    """
    Engineer some interesting timeseries features from the already existing dataframe.
    """

    # create a copy of the data frame as we do not want the original data to be affected
    df = df.copy()

    # add some basic columns
    df['date'] = df.index.date
    df['date'] = pd.to_datetime(df['date'])
    df['time_utc'] = df.index.tz_localize('UTC').time
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week
    df['dayOfWeek'] = df.index.dayofweek
    df['hour_utc'] = df.index.tz_localize('UTC').hour
    df['minute'] = df.index.minute

    # mapping months to seasons: 1- winter, 2-spring, 3-summer, 4-autumn
    if seasons:
        # maps months to seasons:
        seasons = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3,
                   7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}
        df['season'] = df['month'].map(seasons, na_action=None)
        df['winter'] = np.where(df['season'] == 1, True, False)
        df['spring'] = np.where(df['season'] == 2, True, False)
        df['summer'] = np.where(df['season'] == 3, True, False)
        df['autumn'] = np.where(df['season'] == 4, True, False)
        df['transitionperiod'] = np.where(
            (df['season'] == 2) | (df['season'] == 4), True, False)

    # mapping to descriptive times of the day
    if weekday:
        df['weekday'] = np.where(df['dayOfWeek'] <= 4, True, False)
        df['weekend'] = np.where(df['dayOfWeek'] >= 5, True, False)

    # add the time of day
    if daytime:
        df['morning'] = np.where(df['hour_utc'].between(
            6, 10, inclusive='left'), True, False)
        df['noon'] = np.where(df['hour_utc'].between(
            10, 14, inclusive='left'), True, False)
        df['afternoon'] = np.where(df['hour_utc'].between(
            14, 18, inclusive='left'), True, False)
        df['evening'] = np.where(df['hour_utc'].between(
            18, 23, inclusive='left'), True, False)
        df['day'] = np.where(df['hour_utc'].between(
            6, 23, inclusive='left'), True, False)
        df['night'] = np.where((df['hour_utc'] >= 23) |
                               (df['hour_utc'] < 6), True, False)

    if not basic:
        df.drop(columns=['date', 'time_utc', 'year', 'month',
                'week', 'dayOfWeek', 'hour_utc', 'minute'], inplace=True)
    return df.reset_index()
