import os
import shapefile
import pandas as pd
from src.data import get_lat_lon

def load_shape_data_file() -> pd.DataFrame:
    shapefile_path = os.path.join(SHAPE_DATA_DIR, "taxi_zones.shp")
    
    st.write("Shapefile Path:", shapefile_path)
    st.write("Files in SHAPE_DATA_DIR:")
    st.write(os.listdir(SHAPE_DATA_DIR))

    sf = shapefile.Reader(shapefile_path)

    fields_name = [field[0] for field in sf.fields[1:]]
    shp_dic = dict(zip(fields_name, list(range(len(fields_name)))))
    attributes = sf.records()

    shp_attr = [dict(zip(fields_name, attr)) for attr in attributes]

    taxi_zone_lookup = pd.DataFrame(shp_attr).join(get_lat_lon(
        sf, shp_dic).set_index("LocationID"), on="LocationID")

    return taxi_zone_lookup
