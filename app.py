import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ============================================
# CACHE FUNCTIONS
# ============================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_tuned_used_car_price_model.pkl")
        return model
    except:
        try:
            model = joblib.load("../models/best_tuned_used_car_price_model.pkl")
            return model
        except:
            st.error("Model not found!")
            return None

@st.cache_resource
def load_features():
    try:
        features = joblib.load("model_features.pkl")
        return features
    except:
        try:
            features = joblib.load("../models/model_features.pkl")
            return features
        except:
            return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("../data/vehicle_price_prediction.csv")
    except:
        try:
            df = pd.read_csv("data/vehicle_price_prediction.csv")
        except:
            df = pd.read_csv("vehicle_price_prediction.csv")
    
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    categorical_cols = ['make', 'model', 'transmission', 'fuel_type', 'drivetrain',
                        'body_type', 'exterior_color', 'interior_color',
                        'accident_history', 'seller_type', 'condition', 'trim']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    
    return df

@st.cache_data
def get_makes(_df):
    return sorted(_df['make'].dropna().unique())

@st.cache_data
def get_models(_df, make):
    return sorted(_df[_df['make'] == make]['model'].dropna().unique())

# ============================================
# LOAD EVERYTHING
# ============================================

with st.spinner("Loading AI Model..."):
    model = load_model()
    model_features = load_features()
    df = load_data()
    makes = get_makes(df)

st.set_page_config(page_title="AI Used Car Price Predictor", layout="wide")
st.title("🚗 AI Used Car Price Predictor")
st.write("Enter vehicle details to get fair market price")

# ============================================
# GET ALL OPTIONS
# ============================================

transmissions = sorted(df['transmission'].dropna().unique())
fuel_types = sorted(df['fuel_type'].dropna().unique())
drivetrains = sorted(df['drivetrain'].dropna().unique())
body_types = sorted(df['body_type'].dropna().unique())
exterior_colors = sorted(df['exterior_color'].dropna().unique())
interior_colors = sorted(df['interior_color'].dropna().unique())
accident_histories = sorted(df['accident_history'].dropna().unique())
seller_types = sorted(df['seller_type'].dropna().unique())
conditions = sorted(df['condition'].dropna().unique())
trims = sorted(df['trim'].dropna().unique())

# ============================================
# SIDEBAR INPUTS - WITH UNIQUE KEYS
# ============================================

st.sidebar.header("🚗 Vehicle Information")

# Make (with unique key)
selected_make = st.sidebar.selectbox("Make", makes, key="make_select")

# Model (filtered by make, with unique key)
available_models = get_models(df, selected_make)
selected_model = st.sidebar.selectbox("Model", available_models, key="model_select")

# Trim (with unique key)
selected_trim = st.sidebar.selectbox("Trim", trims, key="trim_select")

# Year
selected_year = st.sidebar.slider("Year", 2000, 2025, 2018, key="year_slider")

# Mileage
selected_mileage = st.sidebar.number_input("Mileage (miles)", min_value=0, max_value=300000, value=50000, step=1000, key="mileage_input")

# Engine HP
selected_engine_hp = st.sidebar.number_input("Engine Horsepower", min_value=50, max_value=800, value=200, step=10, key="hp_input")

# Owner count
selected_owner_count = st.sidebar.number_input("Number of Owners", min_value=1, max_value=10, value=1, step=1, key="owner_input")

# Other categorical inputs (all with unique keys)
selected_transmission = st.sidebar.selectbox("Transmission", transmissions, key="transmission_select")
selected_fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types, key="fuel_select")
selected_drivetrain = st.sidebar.selectbox("Drivetrain", drivetrains, key="drivetrain_select")
selected_body_type = st.sidebar.selectbox("Body Type", body_types, key="body_select")
selected_exterior_color = st.sidebar.selectbox("Exterior Color", exterior_colors, key="exterior_select")
selected_interior_color = st.sidebar.selectbox("Interior Color", interior_colors, key="interior_select")
selected_accident_history = st.sidebar.selectbox("Accident History", accident_histories, key="accident_select")
selected_seller_type = st.sidebar.selectbox("Seller Type", seller_types, key="seller_select")
selected_condition = st.sidebar.selectbox("Condition", conditions, key="condition_select")

# Brand popularity
selected_brand_popularity = st.sidebar.slider("Brand Popularity", 1, 100, 50, key="brand_slider")

# ============================================
# PREDICTION
# ============================================

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔮 Predict Price", use_container_width=True, key="predict_btn"):
        start_time = time.time()
        
        # Feature engineering
        current_year = 2025
        vehicle_age = current_year - selected_year
        if vehicle_age == 0:
            vehicle_age = 1
        
        mileage_per_year = selected_mileage / vehicle_age
        hp_per_mileage = selected_engine_hp / (selected_mileage + 1)
        
        # Create input
        input_data = {
            'year': selected_year,
            'mileage': selected_mileage,
            'engine_hp': selected_engine_hp,
            'owner_count': selected_owner_count,
            'vehicle_age': vehicle_age,
            'mileage_per_year': mileage_per_year,
            'brand_popularity': selected_brand_popularity / 100,
            'hp_per_mileage': hp_per_mileage,
            'make': selected_make,
            'model': selected_model,
            'transmission': selected_transmission,
            'fuel_type': selected_fuel_type,
            'drivetrain': selected_drivetrain,
            'body_type': selected_body_type,
            'exterior_color': selected_exterior_color,
            'interior_color': selected_interior_color,
            'accident_history': selected_accident_history,
            'seller_type': selected_seller_type,
            'condition': selected_condition,
            'trim': selected_trim
        }
        
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        
        # Align features
        if model_features:
            for col in model_features:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_features]
        
        # Predict
        prediction = model.predict(input_encoded)[0]
        
        end_time = time.time()
        prediction_time = (end_time - start_time) * 1000
        
        st.markdown("---")
        st.subheader("💰 Predicted Used Car Price")
        
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;">
            <h1 style="color: #2ecc71; font-size: 48px;">${prediction:,.2f}</h1>
            <p style="color: gray;">Estimated fair market value</p>
            <p style="color: gray; font-size: 12px;">⚡ Prediction took: {prediction_time:.0f} milliseconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"📊 Based on {len(df):,} similar vehicles in our database")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "AI Used Car Price Predictor | Final Year Project 2026"
    "</div>",
    unsafe_allow_html=True
)