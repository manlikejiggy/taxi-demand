import numpy as np
import pandas as pd
import holidays
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FunctionTransformer

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool


def average_rides_last_4_hours(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides from
    - 1 hour ago
    - 2 hour ago
    - 3 hour ago
    - 4 hour ago
    """
    X['average_rides_last_4_hours'] = 0.25*(
        X[f'rides_previous_{1}_hour'] +
        X[f'rides_previous_{2}_hour'] +
        X[f'rides_previous_{3}_hour'] +
        X[f'rides_previous_{4}_hour']
    )
    return X


def is_holiday(X: pd.DataFrame) -> pd.DataFrame:
    """
    Appends a column 'is_holiday' to the DataFrame indicating if 'pickup_hour' is a US public holiday.
    """
    us_holidays = holidays.US()
    X_ = X.copy()
    X_['is_holiday'] = X_['pickup_hour'].apply(
        lambda x: x in us_holidays).astype('object')

    return X_


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn data transformation that adds 2 columns
    - hour
    - day_of_week
    and removes the `pickup_hour` datetime column.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_ = X.copy()

        # Generate numeric columns from datetime
        X_["hour"] = X_['pickup_hour'].dt.hour
        X_["day_of_week"] = X_['pickup_hour'].dt.dayofweek

        return X_.drop(columns=['pickup_hour'], axis=1)


def get_pipeline(df: pd.DataFrame) -> Pipeline:

    # Numeric features
    num_col = [col for col in df.columns if col not in ['pickup_location_id', 'pickup_hour', 'icon']]

    num_col = num_col + ['hour', 'day_of_week', 'average_rides_last_4_hours']

    # Categorical Features
    cat_col = ['pickup_location_id', 'is_holiday', 'icon']

    # Preprocess the numerical features
    numerical_processor = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='median')),
        ('num_scaler', MinMaxScaler())
    ])

    # Preprocess the categorical features
    categorical_processor_1 = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('cat_encoder_1', TargetEncoder(target_type="continuous", random_state=42))
    ])

    categorical_processor_2 = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('cat_encoder_2', OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine all data preprocessors with ColumnTransformer
    data_preprocessor = ColumnTransformer(transformers=[
        ('numeric_pre', numerical_processor, num_col),
        ('cat_pre_1', categorical_processor_1, ['is_holiday']),
        ('cat_pre_2', categorical_processor_2, ['pickup_location_id', 'icon'])
    ],
        remainder='passthrough')

    # Function Transformer To Create A New Feature: is_holiday
    Temporal_FE_1 = FunctionTransformer(is_holiday, validate=False)

    # Preprocess the datetime feature to engineer hour & day_of_week
    Temporal_FE_2 = TemporalFeaturesEngineer()

    # Function Transformer To Create A New Feature: average_rides_last_4_weeks
    Temporal_FE_3 = FunctionTransformer(
        average_rides_last_4_hours, validate=False)

    ### PIPELINE ###
    ################

    # Wrap all desired data transformers into the end into Pipeline
    pipeline = Pipeline([
        ('is_holiday', Temporal_FE_1),
        ('engineer_hour_dayofweek', Temporal_FE_2),
        ('avg_rides_last_4_weeks', Temporal_FE_3),
        ('data_preprocessing', data_preprocessor)
    ])

    return pipeline


def get_pipeline_2(df: pd.DataFrame, model) -> Pipeline:

    # Categorical Features
    cat_col = ['icon', 'RatecodeID', 'payment_type',
               'trip_type', 'pickup_location_id']

    # Numeric features
    num_col = [col for col in df.columns if col not in cat_col+['total_amount']]

    # Preprocess the numerical features
    numerical_processor = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='median')),
        ('num_scaler', MinMaxScaler())
    ])

    # Preprocess the categorical features
    categorical_processor_1 = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('cat_encoder_1', OneHotEncoder(handle_unknown='ignore'))
    ])

    categorical_processor_2 = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('cat_encoder_2', TargetEncoder(target_type="continuous", random_state=42))
    ])

    # Combine all data preprocessors with ColumnTransformer
    data_processor = ColumnTransformer(transformers=[
        ('num_preprocessor', numerical_processor, num_col),
        ('cat_preprocessor_1', categorical_processor_1,['icon', 'payment_type', 'trip_type']),
        ('cat_preprocessor_2', categorical_processor_2,['RatecodeID', 'pickup_location_id'])
    ], remainder='passthrough')

    ### PIPELINE ###
    ################

    # Wrap all desired data transformers into the end into Pipeline
    pipeline = Pipeline(steps=[
        ('data_preprocessing', data_processor),
        ('model', model)
    ])

    return pipeline


def track_experiment(experiment_name: str, eval_metrics: dict, model, params: dict = None, tag: str = None) -> None:
    """
    Logs an experiment run with specified metrics and model to MLflow.

    Parameters:
    - experiment_name (str): The name of the experiment under which to log the run.
    - eval_metrics (dict): A dictionary of evaluation metrics to log.
    - model: The model object to log.
    - params (dict, optional): A dictionary of parameters used for the model.

    Returns:
    None
    """

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        if params is not None:
            mlflow.log_params(params)

        mlflow.log_metrics(eval_metrics)
        mlflow.sklearn.log_model(model, "model")

        mlflow.set_tag("tag1", tag)

    print(
        f'Experiment has successfully been logged to {experiment_name} in MLFlow')


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                y_test: pd.Series, name: str, track: bool = True):
    """
    Train The Model
    """

    pipeline = get_pipeline(X_train)

    X_train = pipeline.fit_transform(X_train, np.log1p(y_train))
    X_test = pipeline.transform(X_test)

    y_train = np.log1p(y_train)

    if name.lower() == 'catboost':

        cat_train_data = Pool(data=X_train, label=y_train)
        cat_test_data = Pool(data=X_test, label=np.log1p(y_test))

        model.fit(cat_train_data, eval_set=cat_test_data)

    else:
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.expm1(preds)

    MAE = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {MAE:.2f} | R2-score: {r2:.2f}")
    print("______" * 10, '\n')

    model_result = {"MAE": round(MAE, 2), "R2-Score": round(r2, 2)}

    if track:
        try:
            track_experiment('Taxi-Demand Forecast', model_result,
                             model, model.get_params(), name + ' Taxi-Demand Forecast')
        except:
            print(
                'Cannot log experiemt. Ensure MLFlow backend store URI is set up and working properly')

    return np.round(preds), model_result


def train_price_model(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                      y_test: pd.Series, name: str, track: bool = True):
    """
    Train The Model
    """
    pipeline = get_pipeline_2(X_train, model)

    pipeline.fit(X_train, np.log1p(y_train))

    preds = pipeline.predict(X_test)
    preds = np.round(np.expm1(preds), 2)

    MAE = mean_absolute_error(y_test, preds)
    RMSE = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(
        f"MAE: {MAE:.2f} | RMSE: {RMSE:.2f} | R2-score: {r2:.2f}")
    print("______" * 10, '\n')

    model_result = {'MAE': round(MAE, 2), "RMSE": round(
        RMSE, 2), "R2-Score": round(r2, 2)}

    if track:
        try:
            track_experiment('Dynamic Pricing', model_result,
                             model, model.get_params(), name+' Dynamic Price')
        except:
            print(
                'Cannot log experiemt. Ensure MLFlow backend store URI is set up and working properly')

    return preds, model_result
