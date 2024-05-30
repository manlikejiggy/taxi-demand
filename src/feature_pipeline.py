import pandas as pd 
from datetime import datetime, timedelta
import hopsworks

import src.config as config
from src.data import transform_raw_data_into_ts_data

current_date = pd.to_datetime(datetime.now(), utc=True).floor('h') 
print(f'{current_date=}')

# we fetch raw data for the last 28 days, to add redundancy to our data pipeline
fetch_data_to = current_date
fetch_data_from = current_date - timedelta(days=3) # tried 3 days


from src.data import fetch_ride_events_from_data_warehouse, fetch_batch_raw_ride_data

rides = fetch_batch_raw_ride_data(from_date=fetch_data_from, to_date=fetch_data_to)
ts_data = transform_raw_data_into_ts_data(rides)


# Connect to the project
project = hopsworks.login(project = config.HOPSWORKS_PROJECT_NAME, 
                          api_key_value = config.HOPSWORKS_API_KEY
                          )

# Connect to the feature store
feature_store = project.get_feature_store()

# Connect to the feature group
ride_feature_group = feature_store.get_or_create_feature_group(
    name = config.RIDE_FEATURE_GROUP_NAME,
    version = config.RIDE_FEATURE_GROUP_VERSION,
    description = "Time-series data at hourly frequency",
    primary_key = ['pickup_location_id', 'pickup_hour'],
    event_time = 'pickup_hour',
)



ride_feature_group.insert(ts_data, write_options={"wait_for_job": True})


print("Successfully Inserted Latest Features Into Feature Store")

