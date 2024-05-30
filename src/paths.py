import os
from pathlib import Path

PARENT_DIR = Path(__file__).parent.resolve().parent

DATA_DIR = PARENT_DIR / 'data'
RIDES_DATA_DIR = PARENT_DIR / 'data' / 'rides'
WEATHER_DATA_DIR = PARENT_DIR / 'data' / 'weather'
SHAPE_DATA_DIR = PARENT_DIR / 'data' / 'shape'

RAW_DATA_DIR = PARENT_DIR / 'data' / 'rides' / 'raw'
TRANSFORMED_DATA_DIR = PARENT_DIR / 'data' / 'rides' / 'transformed'

RAW_WEATHER_DATA_DIR = WEATHER_DATA_DIR / 'raw'

MODELS_DIR = PARENT_DIR / 'models'


if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RIDES_DATA_DIR).exists():
    os.mkdir(RIDES_DATA_DIR)

if not Path(WEATHER_DATA_DIR).exists():
    os.mkdir(WEATHER_DATA_DIR)

if not Path(SHAPE_DATA_DIR).exists():
    os.mkdir(SHAPE_DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)
