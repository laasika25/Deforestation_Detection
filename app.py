import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Configuration
PAGE_CONFIG = {"page_title": "Fire Type Classifier", "page_icon": "🔥", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)

# Load model and scaler
@st.cache_data
def load_artifacts():
    try:
        model = joblib.load("best_fire_detection_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

model, scaler = load_artifacts()

# Fire type mappings
FIRE_TYPES = {
    0: ("Vegetation Fire", "🌿", "#4CAF50", "forests, grasslands or agricultural lands"),
    2: ("Static Land Source", "🏭", "#607D8B", "industrial areas or fixed infrastructure"),
    3: ("Offshore Fire", "🛳️", "#2196F3", "water or offshore locations")
}

# CSS styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #FF5722;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #E64A19;
        transform: scale(1.05);
    }
    .stNumberInput div[data-baseweb="input"] {
        border-radius: 8px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🔥 Fire Type Classifier")
st.markdown("""
<div style="background-color: #FFF3E0; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
    <h3 style="color: #E65100;">Predict fire type based on MODIS satellite readings</h3>
    <p style="color: #555;">Enter the satellite observation parameters below to classify the fire source.</p>
</div>
""", unsafe_allow_html=True)

# Input columns
col1, col2 = st.columns(2)

with col1:
    brightness = st.number_input(
        "Brightness (Kelvin)",
        min_value=0.0,
        max_value=500.0,
        value=300.0,
        help="Brightness temperature in Kelvin at channel 21/22"
    )
    
    bright_t31 = st.number_input(
        "Brightness T31 (Kelvin)",
        min_value=0.0,
        max_value=500.0,
        value=290.0,
        help="Brightness temperature in Kelvin at channel 31"
    )
    
    frp = st.number_input(
        "Fire Radiative Power (MW)",
        min_value=0.0,
        max_value=500.0,
        value=15.0,
        help="Fire radiative power in megawatts"
    )

with col2:
    scan = st.number_input(
        "Scan",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Size of the pixel along the scan direction"
    )
    
    track = st.number_input(
        "Track",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Size of the pixel along the track direction"
    )
    
    confidence = st.selectbox(
        "Confidence Level",
        ("low", "nominal", "high"),
        help="Quality confidence of the detection"
    )

confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

if st.button("Classify Fire Type"):
    try:
        # Prepare input
        input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
        # Get result details
        fire_name, fire_emoji, fire_color, fire_desc = FIRE_TYPES.get(
            prediction, 
            ("Unknown", "❓", "#9E9E9E", "unclassified source")
        )
        
        # Show result
        st.markdown(f"""
        <div style="background-color: {fire_color}20; 
                    border-left: 5px solid {fire_color};
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin-top: 1rem;
                    color: {fire_color};">
            <h2 style="margin-bottom: 0.5rem;">{fire_emoji} {fire_name}</h2>
            <p style="color: #333;">This fire detection most likely comes from {fire_desc}.</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("""
<div style="margin-top: 2rem; padding: 1rem; background-color: #F5F5F5; border-radius: 8px;">
    <p style="text-align: center; color: #666;">Fire Type Classification Model | Powered by MODIS satellite data</p>
</div>
""", unsafe_allow_html=True)
