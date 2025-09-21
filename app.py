import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time # New import for simulating a loading effect

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Exoplanet Detection AI",
    page_icon="🪐",
    layout="wide"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('exoplanet_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'exoplanet_model.joblib' is in the correct folder.")
        return None

model = load_model()
if model is None:
    st.stop()

# --- HEADER ---
st.title('Exoplanet Detection AI 🪐')
st.write("This app uses a Machine Learning model to predict whether a celestial object is an exoplanet. Adjust the features in the sidebar to describe a potential planet, then click **'Analyze Celestial Object'** to see the AI's prediction.")

# --- SIDEBAR (USER INPUT) ---
st.sidebar.header('Input Features')

def user_input_features():
    # ... (all the sidebar slider and number_input code is the same) ...
    koi_fpflag_nt = st.sidebar.slider('Not Transit-Like Flag', 0, 1, 0)
    koi_fpflag_ss = st.sidebar.slider('Stellar Eclipse Flag', 0, 1, 0)
    koi_fpflag_co = st.sidebar.slider('Centroid Offset Flag', 0, 1, 0)
    koi_fpflag_ec = st.sidebar.slider('Ephemeris Contamination Flag', 0, 1, 0)
    koi_period = st.sidebar.number_input('Orbital Period [days]', value=9.48)
    koi_time0bk = st.sidebar.number_input('Transit Epoch [BKJD]', value=170.53)
    koi_impact = st.sidebar.number_input('Impact Parameter', value=0.146)
    koi_duration = st.sidebar.number_input('Transit Duration [hrs]', value=2.95)
    koi_depth = st.sidebar.number_input('Transit Depth [ppm]', value=615.8)
    koi_prad = st.sidebar.number_input('Planetary Radius [Earth radii]', value=2.26)
    koi_teq = st.sidebar.number_input('Equilibrium Temperature [K]', value=793.0)
    koi_insol = st.sidebar.number_input('Insolation Flux [Earth flux]', value=93.59)
    koi_steff = st.sidebar.number_input('Stellar Effective Temperature [K]', value=5455.0)
    koi_slogg = st.sidebar.number_input('Stellar Surface Gravity', value=4.467)
    koi_srad = st.sidebar.number_input('Stellar Radius [Solar radii]', value=0.927)
    ra = st.sidebar.number_input('Right Ascension', value=291.93)
    dec = st.sidebar.number_input('Declination', value=48.14)
    koi_kepmag = st.sidebar.number_input('Kepler-band Magnitude', value=15.714)
    
    data = {
        'koi_fpflag_nt': koi_fpflag_nt, 'koi_fpflag_ss': koi_fpflag_ss,
        'koi_fpflag_co': koi_fpflag_co, 'koi_fpflag_ec': koi_fpflag_ec,
        'koi_period': koi_period, 'koi_time0bk': koi_time0bk,
        'koi_impact': koi_impact, 'koi_duration': koi_duration,
        'koi_depth': koi_depth, 'koi_prad': koi_prad, 'koi_teq': koi_teq,
        'koi_insol': koi_insol, 'koi_steff': koi_steff,
        'koi_slogg': koi_slogg, 'koi_srad': koi_srad, 'ra': ra, 'dec': dec,
        'koi_kepmag': koi_kepmag
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- MAIN PANEL ---
# Create two columns
col1, col2 = st.columns([1, 1])

# Column 1: Display User Input in a styled box
with col1:
    st.subheader('Input Features')
    st.dataframe(input_df.T.rename(columns={0: 'Value'}))

# Column 2: The Prediction Area
with col2:
    st.subheader('Model Prediction')
    # --- NEW: Moved the button here and made it more prominent ---
    if st.button('Analyze Celestial Object', type="primary", use_container_width=True):
        # --- NEW: Added a spinner for a loading effect ---
        with st.spinner('Our AI is analyzing the cosmos...'):
            time.sleep(1) # Simulate a short delay for effect
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            confidence = prediction_proba.max()

        if prediction[0] == 'CONFIRMED':
            st.markdown(f"<h2 style='text-align: center; color: green;'>✅ CONFIRMED Exoplanet</h2>", unsafe_allow_html=True)
            # --- NEW: Celebratory balloons for a positive result! ---
            st.balloons()
        else:
            st.markdown(f"<h2 style='text-align: center; color: red;'>❌ FALSE POSITIVE</h2>", unsafe_allow_html=True)
        
        st.write("---")
        # --- NEW: Use st.metric for a dashboard look ---
        st.metric(label="**Prediction Confidence**", value=f"{confidence * 100:.2f}%")

        with st.expander("Show Feature Importance"):
            importances = model.feature_importances_
            feature_names = input_df.columns
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=True)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(feature_importance_df['feature'], feature_importance_df['importance'])
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Model Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Adjust the features in the sidebar and click 'Analyze' to see the result.")