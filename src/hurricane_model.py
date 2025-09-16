import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def convert_lat(val):
    val = val.strip()
    if val.endswith('N'):
        return float(val[:-1])
    elif val.endswith('S'):
        return -float(val[:-1])
    return float(val)

def convert_lon(val):
    val = val.strip()
    if val.endswith('E'):
        return float(val[:-1])
    elif val.endswith('W'):
        return -float(val[:-1])
    return float(val)

def train_hurricane_model(data_path):
    # Load data
    df = pd.read_csv(data_path)
    # Drop rows with missing values in relevant columns
    df = df.dropna(subset=['Maximum Wind', 'Minimum Pressure', 'Latitude', 'Longitude'])
    # Convert Latitude and Longitude to numeric
    df['Latitude'] = df['Latitude'].apply(convert_lat)
    df['Longitude'] = df['Longitude'].apply(convert_lon)
    # Define features and target
    X = df[['Latitude', 'Longitude', 'Minimum Pressure']]
    y = df['Maximum Wind']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    # Save model
    joblib.dump(model, 'hurricane_model.joblib')
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R2 Score: {r2:.4f}')

if __name__ == '__main__':
    train_hurricane_model('data/atlantic.csv')