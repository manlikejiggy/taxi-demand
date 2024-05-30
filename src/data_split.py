from datetime import datetime
from typing import Tuple

import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits a DataFrame into training and testing sets based on a cutoff date.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features and target.
    - cutoff_date (datetime): The date before which data is used for training.
    - target_column_name (str): The name of the target variable column.

    Returns:
    - X_train (pd.DataFrame): Features for the training set.
    - y_train (pd.Series): Target variable for the training set.
    - X_test (pd.DataFrame): Features for the testing set.
    - y_test (pd.Series): Target variable for the testing set.
    """

    train_data = df[df['pickup_hour'] < cutoff_date].reset_index(drop=True)
    test_data = df[df['pickup_hour'] >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]

    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test
