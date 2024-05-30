from datetime import datetime, timedelta
import pandas as pd

from src.inference import load_batch_of_rides_features_from_store
from src.inference import load_batch_of_weather_features_from_store 
from src.inference import prepare_spatio_temporal_data_for_prediction
from src.inference import load_model
from src.inference import get_model_prediction
from src.feature_store import get_feature_store
import src.config as config 


current_date = pd.to_datetime(datetime.now(), utc=True).floor('h')
print(current_date)

# Load Rides & Weather Data From Hopsworks Feature Store

rides = load_batch_of_rides_features_from_store(current_date)
rides.head()

weather = load_batch_of_weather_features_from_store(current_date)
weather.head() 

#Create A Spatio-Temporal Dataset & Prepare Data For Prediction 
X_test = prepare_spatio_temporal_data_for_prediction(rides, weather)


## Get Model

model = load_model(source='local')
predictions_df = get_model_prediction(model, X_test)
predictions_df['pickup_hour'] = current_date


# Save Predictions In Feature Store, So They Can Be Consumed By Web App Later

# Connect to the feature group
feature_group = get_feature_store().get_or_create_feature_group(
    name = config.MODEL_PREDICTIONS_FEATURE_GROUP, 
    version = config.MODEL_PREDICTIONS_FEATURE_GROUP_VERSION,
    description= "Predictions generated by our production model",
    primary_key = ['pickup_location_id', 'pickup_hour'],
    event_time= 'pickup_hour'
)


feature_group.insert(predictions_df, write_options={"wait_for_job": False})


print('Preditions Successfully Inserted To Feature Store')
