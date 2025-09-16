from sklearn.ensemble import RandomForestRegressor
import joblib


def train_model(X_train, y_train):
    """
    Trains a RandomForestRegressor and saves the model to 'wildfire_model.joblib'.
    Args:
        X_train: Training features
        y_train: Training target
    Returns:
        model: Trained RandomForestRegressor
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'wildfire_model.joblib')
    return model 