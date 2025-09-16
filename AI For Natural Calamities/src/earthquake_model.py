import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_earthquake_model(data_path):
    # Load data
    df = pd.read_csv(data_path)
    # Keep only relevant columns and drop missing values
    df = df[['Latitude', 'Longitude', 'Depth', 'Magnitude']].dropna()
    # Bin Magnitude into 4 classes
    bins = [5.5, 6.0, 7.0, 8.0, float('inf')]
    labels = ['Light', 'Moderate', 'Strong', 'Great']
    df['Magnitude_Class'] = pd.cut(df['Magnitude'], bins=bins, labels=labels, right=False)
    # Remove rows with NaN in Magnitude_Class (i.e., Magnitude < 5.5)
    df = df.dropna(subset=['Magnitude_Class'])
    # Define features and target
    X = df[['Latitude', 'Longitude', 'Depth']]
    y = df['Magnitude_Class']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Save model
    joblib.dump(model, 'earthquake_model.joblib')
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_earthquake_model('data/database.csv') 