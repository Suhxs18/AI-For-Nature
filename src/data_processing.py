import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_process_data(file_path):
    """
    Loads the forest fires dataset, processes categorical and target columns, and splits into train/test sets.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(file_path)
    # Convert 'month' and 'day' to categorical codes
    df['month'] = df['month'].astype('category').cat.codes
    df['day'] = df['day'].astype('category').cat.codes
    # Log-transform the 'area' column
    df['log_area'] = np.log1p(df['area'])
    # Define features and target
    X = df.drop(['area', 'log_area'], axis=1)
    y = df['log_area']
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test 