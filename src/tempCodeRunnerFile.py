
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
