from typing import Optional, List
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.graph_objects as go

import shapefile
import geopandas as gpd


plt.style.use('rose-pine-moon')


def plot_one_sample(
        features_df: pd.DataFrame,
        targets_df: pd.Series,
        example_id: int,
        predictions: Optional[pd.Series] = None
):
    """
    Plots the historical ride demand for a single example with an option to include predictions.

    Parameters:
    - features_df (pd.DataFrame): DataFrame containing historical feature data.
    - targets_df (pd.Series): Series containing the target variable (actual values).
    - example_id (int): Index of the example to plot.
    - predictions (pd.Series, optional): Series containing the predicted values, if available.

    Returns:
    - A Plotly figure object visualizing the historical ride demand leading up to the target value,
      with an optional prediction marker.
    """
    features_ = features_df.iloc[example_id]
    target_ = targets_df.iloc[example_id]

    ts_columns = [
        col for col in features_df.columns if col.startswith('rides_previous_')]
    ts_values = [features_[col] for col in ts_columns] + [target_]
    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
        features_['pickup_hour'],
        freq='h'
    )

    # line plot with past values
    title = f"Pickup Hour = {features_['pickup_hour']}, location_id = {features_['pickup_location_id']}"
    fig = px.line(x=ts_dates, y=ts_values, template='plotly_dark', markers=True, title=title)

    # green dot for the value we wanna predict
    fig.add_scatter(x=ts_dates[-1:],
                    y=[target_],
                    line_color='green',
                    mode='markers',
                    marker_size=10,
                    name='actual value'
                    )
    if predictions is not None:
        # big red X for the predicted value, if passed
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(x=ts_dates[-1:], y=[prediction_],
                        line_color='red', mode='markers', marker_symbol='x',
                        marker_size=15, name='prediction')

    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Number of Rides Demanded')

    return fig


def plot_ts(
    ts_data: pd.DataFrame,
    locations: Optional[List[int]] = None
):
    """
    Plot time-series data
    """
    ts_data_to_plot = ts_data[ts_data['pickup_location_id'].isin(
        locations)] if locations else ts_data['ride_demand']

    fig = px.line(
        ts_data_to_plot,
        x="pickup_hour",
        y="ride_demand",
        color='pickup_location_id',
        template='none',
    )

    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Number of Rides Demanded')

    fig.show()



def plot_demand_forecast(merged_row, prediction, location_id):
    
    # Extract historical features from merged_row
    ts_columns = [col for col in merged_row.index if col.startswith('rides_previous_')]
    ts_values = [merged_row[col] for col in ts_columns]
    
    # Create a range of timestamps for plotting
    ts_dates = pd.date_range(
        start=merged_row['pickup_hour_features'] - timedelta(hours=len(ts_columns)),
        periods=len(ts_values),
        freq='h'
    )
    
    title = f"Pickup Hour = {merged_row['pickup_hour_features']}, location_id = {location_id}"
    fig = px.line(x=ts_dates, y=ts_values, title=title, template='plotly_dark', markers=True)
    
    # Red dot for the predicted value
    fig.add_scatter(x=[merged_row['pickup_hour_features']], 
                    y= [merged_row['predicted_demand']],
                    line_color='red', 
                    mode='markers', 
                    marker_symbol='0',
                    marker_size=10, 
                    name='prediction')

    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Rides Demanded')

    return fig



def plot_nyc_tlc_shapefile(shapefile_path: str):
    gdf = gpd.read_file(shapefile_path)

    # Set up the subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 20), dpi=100)

    # Plot boroughs on the first subplot
    gdf_boroughs = gdf.dissolve(by='borough')
    gdf_boroughs.plot(ax=axs[0], cmap='Set2')
    axs[0].set_title('Boroughs in NYC', fontsize=20)
    axs[0].axis('off')

    # Add borough names
    for idx, row in gdf_boroughs.iterrows():
        axs[0].annotate(text=idx,
                        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                        ha='center',
                        va='center',
                        fontsize=15,
                        color='white')

    # Plot zones on the second subplot
    gdf.plot(ax=axs[1], column='LocationID', cmap='Oranges', )
    axs[1].set_title('Zones in NYC', fontsize=20)
    axs[1].axis('off')

    # Add zone numbers
    for idx, row in gdf.iterrows():
        axs[1].annotate(text=row['LocationID'],
                        xy=(row.geometry.centroid.x,
                            row.geometry.centroid.y),
                        ha='center',
                        va='center',
                        fontsize=10,
                        color='black')

    plt.tight_layout()
    plt.show()


def plot_zone_map(file: shapefile, df: pd.DataFrame, n: int = 3, show_legend: bool = True, cmap: str = "Reds"):
    """
    Renders maps for the top n taxi pickup and drop-off zones using a shapefile.

    Parameters:
    - file: Path to the shapefile.
    - df: DataFrame with LocationID, PUcount, and DOcount columns.
    - n: Number of top zones to display (default is 3).
    - show_legend: Flag to display the legend (default is False).

    The maps highlight zones with the highest counts and annotate them with zone names. 
    Assumes 'LocationID' 132 is 'JFK Airport' for any missing 'zone' labels.
    """

    # Sort and get the top 3 locations for pickups and drop-offs
    top_pickups = df.sort_values(
        by='PUcount', ascending=False).head(n)
    top_dropoffs = df.sort_values(
        by='DOcount', ascending=False).head(n)

    gdf = gpd.read_file(file)

    # Convert the GeoDataFrame to the appropriate projected CRS
    gdf = gdf.to_crs(epsg=3857)
    gdf['LocationID'] = gdf['LocationID'].astype(int)
    top_pickups['LocationID'] = top_pickups['LocationID'].astype(int)
    top_dropoffs['LocationID'] = top_dropoffs['LocationID'].astype(int)

    # Merge the top pickups and dropoffs with the GeoDataFrame
    gdf = gdf.merge(
        top_pickups[['LocationID', 'zone', 'PUcount']], on='LocationID', how='left')
    gdf = gdf.merge(
        top_dropoffs[['LocationID', 'zone', 'DOcount']], on='LocationID', how='left')

    # Fill NaN values with 0 for proper plotting
    gdf['PUcount'].fillna(0, inplace=True)
    gdf['DOcount'].fillna(0, inplace=True)

    # Replace 'NaN' in 'zone' column with 'JFK Airport' where 'LocationID' is 132
    gdf.loc[gdf['LocationID'] == 132, 'zone'] = gdf.loc[gdf['LocationID']
                                                        == 132, 'zone'].fillna('JFK Airport')

    # Create the figure and axes for the subplot
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Plot pickups
    gdf.plot(column='PUcount', cmap=cmap, legend=show_legend,
             ax=ax[0], legend_kwds={'label': "Pickup Count"})
    ax[0].set_title('Zones With Most Pickups', fontsize=18)
    ax[0].axis('off')

    # Plot dropoffs
    gdf.plot(column='DOcount', cmap=cmap, legend=show_legend,
             ax=ax[1], legend_kwds={'label': "Dropoff Count"})
    ax[1].set_title('Zones With Most Drop-offs', fontsize=18)
    ax[1].axis('off')

    # Font settings for annotations
    font_size = 10
    font_color = 'yellow'
    bbox_color = 'black'
    bbox_alpha = 0.6

    # Loop through the geodataframe and add annotations
    for idx, row in gdf.iterrows():
        if row['PUcount'] > 0 and not pd.isna(row['zone']):
            centroid = row.geometry.centroid
            ax[0].annotate(text=row['zone'], xy=(centroid.x, centroid.y),
                           xytext=(3, 3), textcoords='offset points',
                           ha='center', fontsize=font_size, color=font_color,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=bbox_alpha))

        if row['DOcount'] > 0 and not pd.isna(row['zone']):
            centroid = row.geometry.centroid
            ax[1].annotate(text=row['zone'], xy=(centroid.x, centroid.y),
                           xytext=(3, 3), textcoords='offset points',
                           ha='center', fontsize=font_size, color=font_color,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=bbox_alpha))

    plt.tight_layout()
    plt.show()


def plot_zone_maps(gdf: gpd.GeoDataFrame, column: str, title: str, ax: list, top_zones: pd.Series):
    """
    Plots a map with annotated top zones.

    Parameters:
    gdf : GeoDataFrame with geographical data.
    column : String indicating the data column for zone colors.
    title : String for the map title.
    ax : Axes object for plotting.
    top_zones : List of LocationIDs for top zones to annotate.
    """
    # Plot the GeoDataFrame
    gdf.plot(column=column, cmap='Reds', linewidth=0.8,
             ax=ax, edgecolor='0.8', legend=True)

    # Set the title of the plot with an increased font size
    ax.set_title(title, fontsize=18)

    # Turn off the axis
    ax.axis('off')

    # Annotate the top zones with text labels
    for loc_id in top_zones:
        row = gdf[gdf['LocationID'] == loc_id]
        if not row.empty:
            x, y = row.geometry.centroid.x, row.geometry.centroid.y

            # Manually adjust the positions for the annotations to prevent overlap
            # Replace `some_id` with the actual ID needing adjustment
            if loc_id in [148, 236, 161]:
                # Adjust these values as needed to prevent overlap
                xytext = (-30, -30)
            else:
                xytext = (30, 30)  # Default offset values

            # Annotate with adjusted offsets
            ax.annotate(text=row['zone'].values[0],
                        xy=(x, y),
                        xytext=xytext,
                        textcoords='offset points',
                        ha='center',
                        fontsize=10,
                        fontweight='bold',
                        arrowprops=dict(arrowstyle="->",
                                        color='black', lw=1.5, alpha=0.6),
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='yellow', boxstyle='round,pad=0.5'))


def plot_borough_heatmaps(file: shapefile, df: pd.DataFrame, pickup_col: str, dropoff_col: str, cmap: str = 'Blues'):
    """
    Generates side-by-side choropleth maps for NYC taxi pickups and drop-offs.

    Parameters:
        file (str): The file path to the boroughs shapefile.
        df (pd.DataFrame): Data with pickup and drop-off counts.
        pickup_col (str): DataFrame column for pickup data.
        dropoff_col (str): DataFrame column for drop-off data.
        cmap (str, optional): Colormap for the maps, default is 'Blues'.

    The function creates a visual comparison between taxi service pickups and drop-offs across NYC boroughs,
    displaying this data on a map with annotated counts.
    """
    gdf = gpd.read_file(file)
    gdf = gdf.merge(df, on='borough')

    # Aggregate geometries by borough
    gdf = gdf.dissolve(by='borough', aggfunc='sum')

    fig, axes = plt.subplots(1, 2, figsize=(25, 12))

    # Plot pickups
    gdf.plot(column=pickup_col, cmap=cmap,
             linewidth=0.8, ax=axes[0], edgecolor='0.8')
    axes[0].set_title('Boroughs with Most Pickups', fontsize=20)
    axes[0].axis('off')
    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=gdf[pickup_col].min(), vmax=gdf[pickup_col].max()))
    sm1._A = []  # workaround for ScalarMappable bug
    cbar1 = fig.colorbar(sm1, ax=axes[0])

    # Plot drop-offs
    gdf.plot(column=dropoff_col, cmap=cmap,
             linewidth=0.8, ax=axes[1], edgecolor='0.8')
    axes[1].set_title('Boroughs with Most Drop-offs', fontsize=20)
    axes[1].axis('off')
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=gdf[dropoff_col].min(), vmax=gdf[dropoff_col].max()))
    sm2._A = []  # workaround for ScalarMappable bug
    cbar2 = fig.colorbar(sm2, ax=axes[1])

    # Add labels onto the maps
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        axes[0].annotate(text='{}\n({:.2f}K)'.format(idx, row[pickup_col]/1000), xy=(centroid.x, centroid.y),
                         horizontalalignment='center', fontsize=15, verticalalignment='center',
                         bbox=dict(facecolor='black', alpha=0.8, edgecolor='none'), color='white')
        axes[1].annotate(text='{}\n({:.2f}K)'.format(idx, row[dropoff_col]/1000), xy=(centroid.x, centroid.y),
                         horizontalalignment='center', fontsize=15, verticalalignment='center',
                         bbox=dict(facecolor='black', alpha=0.8, edgecolor='none'), color='white')

    plt.tight_layout()
    plt.show()


def diff_short_long_trip_on_time(df_pu, df_do):
    # This function will create a 2x2 subplot with the given data.
    def plt_clock(ax, radii, title, color):
        theta = np.linspace(0.0, 2 * np.pi, len(radii), endpoint=False)
        width = 2 * np.pi / len(radii)

        bars = ax.bar(theta, radii, width=width, bottom=0.0,
                      color=color, alpha=0.5, edgecolor='white')

        # Only annotate every third bar to avoid clutter
        for i, (bar, radius) in enumerate(zip(bars, radii)):
            if i % 3 == 0:  # change the modulus to adjust the frequency of annotated bars
                ax.annotate(
                    '{}'.format(radius),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, color='white'
                )

        ax.set_xticks(theta)
        ax.set_xticklabels(['{:02d}:00'.format(hour)
                           for hour in range(24)], fontsize=10)
        ax.set_yticklabels([])
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_facecolor('#343a40')  # Dark background for better contrast
        ax.set_title(title, size=20, color='white')

    # Create the 2x2 subplot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(
        18, 18), subplot_kw={'projection': 'polar'})

    # Use a more aesthetic color palette, e.g., deep red and deep blue
    color_short = 'red'  # '#e63946'  # A deep red color
    color_long = 'blue'  # '#1d3557'   # A deep blue color

    # Plotting each subplot
    plt_clock(axes[0, 0], df_pu['short'],
              "Pickup Time for Short Trips", color_short)
    plt_clock(axes[0, 1], df_pu['long'],
              "Pickup Time for Long Trips", color_long)
    plt_clock(axes[1, 0], df_do['short'],
              "Dropoff Time for Short Trips", color_short)
    plt_clock(axes[1, 1], df_do['long'],
              "Dropoff Time for Long Trips", color_long)

    plt.tight_layout(pad=3)
    plt.show()


def diff_short_long_trip_on(df: pd.DataFrame, col: str, rpr: str = "count", kind: str = 'bar'):
    """
    Visualizes the difference in distribution of an attribute between short and long trips.

    Parameters:
    df (DataFrame): DataFrame containing the ride data.
    col (str): Column name to compare between short and long trips.
    rpr (str): The representation for plotting - "count" or "proportion".
    kind (str): The kind of plot to generate, e.g., 'bar'.
    """
    # Define short and long trips within the DataFrame
    df_short = df[df['trip_distance'] < 30].groupby(
        col).size().reset_index(name='count')
    df_long = df[df['trip_distance'] >= 30].groupby(
        col).size().reset_index(name='count')

    # Calculate proportions if required
    if rpr == "proportion":
        df_short['proportion'] = df_short['count'] / df_short['count'].sum()
        df_long['proportion'] = df_long['count'] / df_long['count'].sum()

    # Merge short and long trips DataFrames on the attribute
    df_merged = pd.merge(df_short, df_long, on=col,
                         how='outer', suffixes=("_short", "_long"))

    # Rename columns for plotting
    df_merged.rename(columns={f'{rpr}_short': 'short trips',
                     f'{rpr}_long': 'long trips'}, inplace=True)

    # Plotting
    ax = df_merged.plot(
        x=col, y=['short trips', 'long trips'], kind=kind, figsize=(15, 6))
    ax.set_ylabel(rpr)
    ax.set_title(
        f'{col.replace("_", " ")} difference in short/long trip'.title(), fontsize=16)

    plt.tight_layout()
    plt.show()
