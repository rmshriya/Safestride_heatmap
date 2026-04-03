import os
import math
import random
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "events_dataset.csv"
MODEL_FILE = "risk_model.pkl"

RISK_LABELS = ["safe", "warning", "emergency"]
RISK_TO_SCORE = {"safe": 1, "warning": 2, "emergency": 3}
SCORE_TO_RISK = {1: "safe", 2: "warning", 3: "emergency"}

FEATURE_COLUMNS = [
    "heel_tap_count",
    "fall_detected",
    "gps_risk_zone",
    "is_night",
    "battery_low",
    "user_moving_fast",
    "hour",
]

ALL_COLUMNS = FEATURE_COLUMNS + ["latitude", "longitude", "risk_level"]


def create_seed_dataset(n=120):
    rows = []
    base_lat = 13.0827
    base_lon = 80.2707

    for _ in range(n):
        hour = random.randint(0, 23)
        is_night = 1 if hour >= 19 or hour <= 5 else 0
        heel_tap_count = random.randint(0, 3)
        fall_detected = random.randint(0, 1)
        gps_risk_zone = random.choices([0, 1, 2], weights=[4, 3, 3])[0]
        battery_low = random.choices([0, 1], weights=[8, 2])[0]
        user_moving_fast = random.choices([0, 1], weights=[6, 4])[0]

        score = 0
        score += heel_tap_count * 1.2
        score += fall_detected * 3.0
        score += gps_risk_zone * 1.5
        score += is_night * 1.0
        score += battery_low * 0.5
        score += user_moving_fast * 1.0

        if score >= 6:
            risk_level = "emergency"
        elif score >= 3:
            risk_level = "warning"
        else:
            risk_level = "safe"

        lat = base_lat + random.uniform(-0.03, 0.03)
        lon = base_lon + random.uniform(-0.03, 0.03)

        rows.append([
            heel_tap_count,
            fall_detected,
            gps_risk_zone,
            is_night,
            battery_low,
            user_moving_fast,
            hour,
            lat,
            lon,
            risk_level,
        ])

    return pd.DataFrame(rows, columns=ALL_COLUMNS)


def load_dataset():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        return df
    df = create_seed_dataset()
    df.to_csv(DATA_FILE, index=False)
    return df


def train_model(df):
    X = df[FEATURE_COLUMNS]
    y = df["risk_level"]
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model


def load_or_train_model(df):
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return train_model(df)
    return train_model(df)


def predict_risk(model, event_dict):
    X = pd.DataFrame([event_dict])[FEATURE_COLUMNS]
    return model.predict(X)[0]


def append_event(df, model, event_row):
    predicted_risk = predict_risk(model, event_row)
    full_row = event_row.copy()
    full_row["risk_level"] = predicted_risk

    new_df = pd.concat([df, pd.DataFrame([full_row])], ignore_index=True)
    new_df.to_csv(DATA_FILE, index=False)

    new_model = train_model(new_df)
    return new_df, new_model, predicted_risk


def build_time_risk_table(df):
    temp = df.copy()
    temp["risk_score"] = temp["risk_level"].map(RISK_TO_SCORE)
    grouped = (
        temp.groupby("hour", as_index=False)["risk_score"]
        .mean()
        .sort_values("hour")
    )
    grouped["time_risk"] = grouped["risk_score"].apply(
        lambda x: "High" if x >= 2.5 else ("Medium" if x >= 1.5 else "Low")
    )
    return grouped


def get_summary_counts(df):
    return {
        "safe": int((df["risk_level"] == "safe").sum()),
        "warning": int((df["risk_level"] == "warning").sum()),
        "emergency": int((df["risk_level"] == "emergency").sum()),
    }


def risk_weight(label):
    return {"safe": 20, "warning": 60, "emergency": 100}[label]


st.set_page_config(page_title="Smart Safety AI Dashboard", layout="wide")
st.title("Smart Safety AI/ML Real-Time Dashboard")

df = load_dataset()
model = load_or_train_model(df)

st.sidebar.header("Add New Event")

heel_tap_count = st.sidebar.selectbox("Heel Tap Count", [0, 1, 2, 3], index=0)
fall_detected = st.sidebar.selectbox("Fall Detected", [0, 1], index=0)
gps_risk_zone = st.sidebar.selectbox("GPS Risk Zone", [0, 1, 2], index=1)
battery_low = st.sidebar.selectbox("Battery Low", [0, 1], index=0)
user_moving_fast = st.sidebar.selectbox("User Moving Fast", [0, 1], index=0)
hour = st.sidebar.slider("Hour of Day", 0, 23, datetime.now().hour)
latitude = st.sidebar.number_input("Latitude", value=13.0827, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=80.2707, format="%.6f")

is_night = 1 if hour >= 19 or hour <= 5 else 0

new_event = {
    "heel_tap_count": heel_tap_count,
    "fall_detected": fall_detected,
    "gps_risk_zone": gps_risk_zone,
    "is_night": is_night,
    "battery_low": battery_low,
    "user_moving_fast": user_moving_fast,
    "hour": hour,
    "latitude": latitude,
    "longitude": longitude,
}

if st.sidebar.button("Predict + Add Event"):
    df, model, predicted_risk = append_event(df, model, new_event)
    st.sidebar.success(f"Predicted Risk: {predicted_risk.upper()}")

counts = get_summary_counts(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", len(df))
col2.metric("Safe", counts["safe"])
col3.metric("Warning", counts["warning"])
col4.metric("Emergency", counts["emergency"])

st.subheader("Latest Events")
st.dataframe(df.tail(10), use_container_width=True)

st.subheader("Heatmap / Risk Map")

map_df = df.copy()
map_df["weight"] = map_df["risk_level"].apply(risk_weight)

layer = pdk.Layer(
    "HeatmapLayer",
    data=map_df,
    get_position="[longitude, latitude]",
    get_weight="weight",
    radiusPixels=60,
    intensity=1,
    threshold=0.2,
)

view_state = pdk.ViewState(
    latitude=float(map_df["latitude"].mean()),
    longitude=float(map_df["longitude"].mean()),
    zoom=11,
    pitch=40,
)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=view_state,
    layers=[layer],
))

st.subheader("Time-Based Risk Scores")
time_risk_df = build_time_risk_table(df)
st.bar_chart(time_risk_df.set_index("hour")["risk_score"])

st.subheader("Hourly Risk Table")
st.dataframe(time_risk_df, use_container_width=True)

st.subheader("How this works")
st.write(
    """
- Each new event is added to the dataset.
- The model retrains on the expanded dataset.
- The heatmap updates from all stored events.
- Time-based risk updates from historical event patterns.
- This makes the demo look live and continuously improving.
"""
)
