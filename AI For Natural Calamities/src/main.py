from data_processing import load_and_process_data
from model_training import train_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def main():
    data_path = 'data/forestfires.csv'
    X_train, X_test, y_train, y_test = load_and_process_data(data_path)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'R2 Score: {r2:.4f}')


if __name__ == '__main__':
    main() 