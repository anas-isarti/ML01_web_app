import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="UFO Sighting Predictor", page_icon="🛸")
st.title("🛸 UFO Sighting Country Predictor")
st.markdown("This frontend calls a FastAPI backend to make predictions.")

API_URL = "http://localhost:8001/predict"

country_names = {
    'us': 'United States', 'ca': 'Canada', 'gb': 'United Kingdom',
    'au': 'Australia', 'de': 'Germany', 'fr': 'France', 'other': 'Other',
}

st.sidebar.header("Sighting Details")

latitude = st.sidebar.slider("Latitude", -90.0, 90.0, 40.0)
longitude = st.sidebar.slider("Longitude", -180.0, 180.0, -90.0)
duration = st.sidebar.slider("Duration (seconds)", 1, 7200, 300)
hour = st.sidebar.slider("Hour of day", 0, 23, 21)
month = st.sidebar.selectbox("Month", range(1, 13), index=6)
day_of_week = st.sidebar.selectbox(
    "Day of week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x],
    index=4
)

if st.sidebar.button("Predict Country", type="primary"):
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "duration_seconds": duration,
        "hour": hour,
        "month": month,
        "day_of_week": day_of_week,
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            st.subheader(
                f"Predicted Country: **:green[{data['predicted_country_name']}]**"
            )
            prob_df = pd.DataFrame(
                list(data['probabilities'].items()),
                columns=['Country', 'Probability']
            ).sort_values('Probability', ascending=False)
            st.bar_chart(prob_df.set_index('Country'), height=300)
        else:
            st.error(f"API error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {API_URL}. Make sure ufo_backend.py is running.")
