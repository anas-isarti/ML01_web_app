import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="UFO Sighting Predictor", page_icon="🛸")
st.title("🛸 UFO Sighting Country Predictor")
st.markdown("Predict which country reported a UFO sighting based on duration and location.")

@st.cache_resource
def load_model():
    model = joblib.load('ufo_model.pkl')
    le = joblib.load('ufo_label_encoder.pkl')
    return model, le

model, le = load_model()

st.sidebar.header("Sighting Details")

latitude = st.sidebar.slider("Latitude", -90.0, 90.0, 40.0)
longitude = st.sidebar.slider("Longitude", -180.0, 180.0, -90.0)
duration = st.sidebar.slider("Duration (seconds)", 1, 7200, 300)
hour = st.sidebar.slider("Hour of day", 0, 23, 21)
month = st.sidebar.selectbox("Month", range(1, 13), index=6)
day_of_week = st.sidebar.selectbox(
    "Day of week", options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x], index=4
)

if st.sidebar.button("Predict Country", type="primary"):
    input_df = pd.DataFrame([{
        'latitude': latitude, 'longitude': longitude,
        'duration_seconds': duration, 'hour': hour,
        'month': month, 'day_of_week': day_of_week,
    }])

    pred_encoded = model.predict(input_df)[0]
    pred_country = le.inverse_transform([pred_encoded])[0]

    country_names = {
        'us': 'United States', 'ca': 'Canada', 'gb': 'United Kingdom',
        'au': 'Australia', 'de': 'Germany', 'fr': 'France', 'other': 'Other',
    }

    st.subheader(f"Predicted Country: **:green[{country_names.get(pred_country, pred_country)}]**")
    st.markdown(f"Confidence: {float(np.max(model.predict_proba(input_df)[0])):.2%}")

    prob_df = pd.DataFrame({
        'Country': [country_names.get(c, c) for c in le.classes_],
        'Probability': model.predict_proba(input_df)[0],
    }).sort_values('Probability', ascending=False)

    for _, row in prob_df.iterrows():
        bar = "█" * int(row['Probability'] * 30)
        st.markdown(f"{row['Country']:15s} {bar} {row['Probability']:.1%}")

st.markdown("---")
st.markdown("""
### About
This app uses a **Random Forest classifier** trained on synthetic UFO sighting data to predict **which country** reported a UFO sighting, based on location, duration, and time features.
""")
