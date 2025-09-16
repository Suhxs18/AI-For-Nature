import streamlit as st
import joblib
import numpy as np

# Load models
wildfire_model = joblib.load('wildfire_model.joblib')
hurricane_model = joblib.load('hurricane_model.joblib')
earthquake_model = joblib.load('earthquake_model.joblib')

st.title('Multi-Disaster Analysis Tool')

# Sidebar selection
option = st.sidebar.selectbox(
    'Choose Analysis Type:',
    ('Wildfire Prediction', 'Hurricane Intensity Prediction', 'Seismic Event Analysis')
)

if option == 'Wildfire Prediction':
    st.header('Wildfire Area Prediction')
    X_coord = st.number_input('X (spatial coordinate)', min_value=1, max_value=9, value=4)
    Y_coord = st.number_input('Y (spatial coordinate)', min_value=2, max_value=9, value=4)
    month = st.number_input('Month (as integer code)', min_value=0, max_value=11, value=5)
    day = st.number_input('Day (as integer code)', min_value=0, max_value=6, value=2)
    FFMC = st.number_input('FFMC', min_value=0.0, max_value=100.0, value=85.0)
    DMC = st.number_input('DMC', min_value=0.0, max_value=300.0, value=100.0)
    DC = st.number_input('DC', min_value=0.0, max_value=900.0, value=500.0)
    ISI = st.number_input('ISI', min_value=0.0, max_value=50.0, value=10.0)
    temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=20.0)
    RH = st.number_input('Relative Humidity (%)', min_value=0.0, max_value=100.0, value=40.0)
    wind = st.number_input('Wind (km/h)', min_value=0.0, max_value=50.0, value=5.0)
    rain = st.number_input('Rain (mm)', min_value=0.0, max_value=50.0, value=0.0)

    if st.button('Predict Wildfire Area (log scale)'):
        features = np.array([[X_coord, Y_coord, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain]])
        log_area_pred = wildfire_model.predict(features)[0]
        area_pred = np.expm1(log_area_pred)
        st.success(f'Predicted Burned Area: {area_pred:.2f} ha (log_area: {log_area_pred:.2f})')

elif option == 'Hurricane Intensity Prediction':
    st.header('Hurricane Maximum Wind Speed Prediction')
    lat = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=25.0)
    lon = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=-80.0)
    min_pressure = st.number_input('Minimum Pressure (mb)', min_value=800.0, max_value=1100.0, value=950.0)
    if st.button('Predict Maximum Wind Speed'):
        features = np.array([[lat, lon, min_pressure]])
        wind_pred = hurricane_model.predict(features)[0]
        st.success(f'Predicted Maximum Wind Speed: {wind_pred:.2f} knots')

else:
    st.header('Seismic Event Magnitude Classification')
    lat = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=0.0)
    lon = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=0.0)
    depth = st.number_input('Depth (km)', min_value=0.0, max_value=700.0, value=10.0)
    if st.button('Classify Magnitude Class'):
        features = np.array([[lat, lon, depth]])
        mag_class = earthquake_model.predict(features)[0]
        st.success(f'Predicted Magnitude Class: {mag_class}') 