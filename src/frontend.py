import numpy as np
import pandas as pd
import requests

import shapefile
import geopandas as gpd
import pydeck as pdk
from datetime import datetime, timedelta

import streamlit as st
import pydeck as pdk
from pyproj import Transformer

from src.inference import load_batch_of_rides_features_from_store, load_rides_prediction_from_store
from src.paths import DATA_DIR, SHAPE_DATA_DIR
from src.data import get_lat_lon
from src.plot import plot_one_sample, plot_demand_forecast

st.write(SHAPE_DATA_DIR)


# Title

current_date = pd.to_datetime(datetime.now(), utc=True).floor('H')
current_date_str = str(current_date.strftime('%Y-%m-%d %H:%M'))
st.title(f'NYC Taxi Demand Prediction 🚖')

# Create the header with HTML
dev = "Made by Princewill. Let's connect 🤝"

# Social Media Handles
twitter = "https://twitter.com/manlikejiggy"
linkedin = "https://www.linkedin.com/in/princewill-nwoko-419794253"
st.markdown(
    f"<href>{dev}</href>"
    f" • <a href='{linkedin}'>LinkedIn</a> • "
    f"<a href='{twitter}'>Twitter</a>",
    unsafe_allow_html=True,
)

st.header(f'{current_date_str} UTC')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 6


def load_shape_data_file() -> pd.DataFrame:
    """
    """
    shapefile_path = f"{SHAPE_DATA_DIR}/taxi_zones.shp"

    sf = shapefile.Reader(shapefile_path)

    fields_name = [field[0] for field in sf.fields[1:]]
    shp_dic = dict(zip(fields_name, list(range(len(fields_name)))))
    attributes = sf.records()

    shp_attr = [dict(zip(fields_name, attr)) for attr in attributes]

    taxi_zone_lookup = pd.DataFrame(shp_attr).join(get_lat_lon(
        sf, shp_dic).set_index("LocationID"), on="LocationID")

    return taxi_zone_lookup


@st.cache_data
def _load_batch_of_rides_features_from_store(current_date: datetime) -> pd.DataFrame:
    """Wrapped version of src.inference.load_batch_of_rides_features_from_store, so
    we can add Streamlit caching

    Args:
        current_date (datetime): _description_

    Returns:
        pd.DataFrame: n_features + 2 columns:
            - `rides_previous_N_hour`
            - `rides_previous_{N-1}_hour`
            - ...
            - `rides_previous_1_hour`
            - `pickup_hour`
            - `pickup_location_id`
    """
    return load_batch_of_rides_features_from_store(current_date)


@st.cache_data
def _load_rides_prediction_from_store(from_pickup_hour: datetime, to_pickup_hour: datetime) -> pd.DataFrame:
    """
    Wrapped version of src.inference.load_rides_prediction_from_store, so we
    can add Streamlit caching

    Args:
        from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 2 columns: pickup_location_id, predicted_demand
    """
    return load_rides_prediction_from_store(from_pickup_hour, to_pickup_hour)


with st.spinner(text="Downloading shapefile to plot NYC Borough"):
    geo_df = load_shape_data_file()
    st.sidebar.write('✅ Shapefile was downloaded')
    progress_bar.progress(1/N_STEPS)


try:
    with st.spinner(text="Fetaching Model Prediction From FeatureStore"):
        predictions_df = _load_rides_prediction_from_store(
            from_pickup_hour=current_date - timedelta(hours=6),
            to_pickup_hour=current_date
        )

        predictions_df = predictions_df.reset_index(drop=True)
        st.sidebar.write("✅ Model Predictions Arrived")
        progress_bar.progress(2/N_STEPS)
        # st.write(predictions_df)

except Exception as e:
    # Error Handling
    st.error(f"An error occurred: {str(e)}")
    st.warning("⚠️ Trying to use older predictions...")

    # Retrying
    st.warning(f"Retrying...")
    with st.spinner(text="Fetching Model Prediction From FeatureStore"):
        predictions_df = _load_rides_prediction_from_store(
            from_pickup_hour=current_date - timedelta(hours=3),
            to_pickup_hour=current_date
        )

        predictions_df = predictions_df.reset_index(drop=True)
        st.sidebar.write("✅ Model Predictions Arrived")
        progress_bar.progress(2/N_STEPS)


# Helper function to check for prediction availability
def check_prediction_availability(preds_df: pd.DataFrame, current_date: datetime, time_offset: int):
    check_time = current_date - timedelta(hours=time_offset)
    return not preds_df[preds_df['pickup_hour'] == check_time].empty


# Helper function to get the most recent available predictions
def get_recent_predictions(preds_df: pd.DataFrame, current_date: datetime, max_offset_hours: int):
    for hour_offset in range(max_offset_hours + 1):
        if check_prediction_availability(preds_df, current_date, hour_offset):
            preds_subset = preds_df[preds_df['pickup_hour'] == (
                current_date - timedelta(hours=hour_offset))]
            if hour_offset > 0:
                st.subheader(
                    f'⚠️ Most Recent Data Is Not Available. Using Last {hour_offset} Hour Predictions')
            return preds_subset, hour_offset

    # If no predictions are available within the offset window, return None
    return None, None


# Main flow of checking predictions
current_date = pd.to_datetime(datetime.now(), utc=True).floor('H')
print("Current time:", current_date)
print("Unique prediction times in DataFrame:",
      predictions_df['pickup_hour'].unique())

predictions_df, hour_offset_found = get_recent_predictions(
    predictions_df, current_date, max_offset_hours=6)

if predictions_df is None:
    st.error(
        "Features aren't available for the last 6 hours. Is the feature pipeline up & running? 🤔")
else:
    predictions_df = predictions_df.reset_index(drop=True)


with st.spinner(text="Preparing Data To Plot"):

    # Function to convert coordinates
    def convert_coords(lat, lon):
        transformer = Transformer.from_crs(
            'epsg:2263', 'epsg:4326', always_xy=True)
        lon, lat = transformer.transform(lon, lat)
        return lat, lon

    # Convert coordinate in the geo_df from Plane to WGS 84 coordinate system
    geo_df['latitude'], geo_df['longitude'] = zip(
        *geo_df.apply(lambda row: convert_coords(row['latitude'], row['longitude']), axis=1))

    # Merging geo_df and predictons_df DataFrames
    geo_pred_df = pd.merge(geo_df, predictions_df,
                           left_on='LocationID', right_on='pickup_location_id')

    # Pseudocolor function for fill colors

    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        f = (val - minval) / (maxval - minval)
        return [int(s + f * (e - s)) for s, e in zip(startcolor, stopcolor)]

    BLACK, ORANGE = (0, 0, 0), (255, 128, 0)

    # Normalize and apply pseudocolor
    max_demand = geo_pred_df['predicted_demand'].max()
    min_demand = geo_pred_df['predicted_demand'].min()
    geo_pred_df['fill_color'] = geo_pred_df['predicted_demand'].apply(
        lambda x: pseudocolor(x, min_demand, max_demand, BLACK, ORANGE))

    progress_bar.progress(3/N_STEPS)


with st.spinner(text="Generating NYC Map"):

    # Set elevation
    geo_pred_df['elevation'] = geo_pred_df['predicted_demand'] * \
        2  # Adjust the multiplier as necessary

    # PyDeck Layer
    layer = pdk.Layer(
        "ColumnLayer",
        geo_pred_df,
        get_position=["longitude", "latitude"],
        get_elevation="elevation",
        get_fill_color="fill_color",
        elevation_scale=1,
        radius=100,
        pickable=True,
        extruded=True,
    )

    # Define initial view state
    view_state = pdk.ViewState(
        latitude=40.7128, longitude=-74.0060, zoom=11, pitch=45, bearing=0)

    # PyDeck Deck
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{zone}\nBorough: {borough}\nPredicted Demand: {predicted_demand}"})

    # Display NYC map
    st.pydeck_chart(r)

    progress_bar.progress(4/N_STEPS)


with st.spinner(text="Fetching Batch of Features Used In The Last Run From FeatureStore"):
    features_df = _load_batch_of_rides_features_from_store(current_date)
    features_df = features_df.reset_index(drop=True)
    st.sidebar.write('✅ Inference Features Fetched')
    progress_bar.progress(5/N_STEPS)


with st.spinner(text="Plotting Time-Series Data"):

    # After loading features_df and predictions_df
    features_df = features_df.reset_index(drop=True)
    predictions_df = predictions_df.reset_index(drop=True)

    # Merge features and predictions dataframe on 'pickup_location_id' for plotting time series data
    merged_df = pd.merge(features_df, predictions_df,
                         on='pickup_location_id', suffixes=('_features', '_preds'))
    merged_df = merged_df.merge(
        geo_df[['LocationID', 'zone']], right_on='LocationID', left_on='pickup_location_id')
    merged_df['predicted_demand'] = np.clip(
        merged_df['predicted_demand'], 0, None)

    # Sort merged_df by 'predicted_demand' in descending order and get the top 10 rows
    top_10_preds = merged_df.sort_values(
        by='predicted_demand', ascending=False).head(10)

    # Add download button in the top right corner
    df_to_download = geo_pred_df.copy().drop(
        ['elevation', 'fill_color', 'LocationID'], axis=1)
    button = st.download_button(
        label="Download Predictions CSV",
        data=df_to_download.to_csv(index=False).encode('utf-8'),
        file_name='taxi_demand_predictions.csv',
        key='download_prediction'
    )

    st.markdown("<div style='text-align: left; font-size: small;'>Note ⚠️: \
                This data should not be used for operational purposes. \
                Data from the most recent hours are not available. Hence, a travel \
                simulation was performed and used as a reliable basis for making forecasts.\
                </div>", unsafe_allow_html=True)

    for index, row in top_10_preds.iterrows():
        location_id = row['pickup_location_id']
        location_name = row['zone']
        prediction = row['predicted_demand']
        max_hour_prediction = row['pickup_hour_preds']

        # Display information in Streamlit
        st.header(f'Zone: {location_name} [Location ID: {location_id}]')
        st.metric(label="Max Rides Predicted", value=int(prediction))

        # Generate and display the plot
        fig = plot_demand_forecast(
            merged_row=row,
            prediction=prediction,
            location_id=location_id
        )
        st.plotly_chart(fig, use_container_width=True)

    progress_bar.progress(6/N_STEPS)
