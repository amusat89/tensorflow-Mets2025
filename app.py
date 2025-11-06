import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os

# Set environment variable to prevent macOS issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(
    page_title="Metabolic Syndrome Predictor",
    page_icon="🏥",
    layout="wide"
)

# Debug: Show current files
st.write("## 🔍 Debug: Checking Files")
current_files = os.listdir('.')
st.write("Files in directory:", [f for f in current_files if f.endswith('.keras')])

# Define feature names
feature_names = [
    'LBDAPBSI', 'BMXBMI', 'BMXWAIST', 'Systolic_BP', 'Diastolic_BP', 
    'RIDAGEYR', 'RIAGENDR', 'DXXSATA', 'DXXSATM', 'DXXVFATA', 
    'DXXVFATM', 'LBXGH', 'LBDGLUSI', 'LBDHDDSI', 'LBXHSCRP', 
    'LBDINSI', 'LBDTCSI', 'LBDTRSI', 'LBDLDLSI', 'eLDL_Trig', 
    'Fasting_hrs'
]

# Load model - FIXED: Only load once
@st.cache_resource
def load_model():
    model_path = 'my_mets_classifier.keras'
    
    # Check if file exists first
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please make sure 'my_mets_classifier.keras' is in the same folder as app.py")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model (ONCE)
model = load_model()

# Stop if model failed to load
if model is None:
    st.stop()

# App title
st.title("🏥 Metabolic Syndrome Prediction")
st.markdown("Predict metabolic syndrome risk using clinical measurements")

# Input form
st.header("Patient Clinical Data")

# Organize inputs
col1, col2, col3 = st.columns(3)

input_data = {}

with col1:
    st.subheader("Basic Info")
    input_data['BMXBMI'] = st.slider("BMI", 15.0, 50.0, 25.0)
    input_data['BMXWAIST'] = st.slider("Waist Circumference (cm)", 50.0, 150.0, 90.0)
    input_data['RIDAGEYR'] = st.slider("Age", 18, 100, 45)
    input_data['RIAGENDR'] = st.radio("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

with col2:
    st.subheader("Blood Pressure")
    input_data['Systolic_BP'] = st.slider("Systolic BP", 80, 200, 120)
    input_data['Diastolic_BP'] = st.slider("Diastolic BP", 40, 120, 80)

with col3:
    st.subheader("Blood Glucose")
    input_data['LBXGH'] = st.slider("HbA1c (%)", 4.0, 15.0, 5.5)
    input_data['LBDGLUSI'] = st.slider("Glucose (mmol/L)", 3.0, 20.0, 5.0)

# Advanced inputs in expander
with st.expander("Advanced Blood Tests"):
    col4, col5 = st.columns(2)
    
    with col4:
        input_data['LBDHDDSI'] = st.slider("HDL Cholesterol", 0.5, 3.0, 1.3)
        input_data['LBDTCSI'] = st.slider("Total Cholesterol", 2.0, 10.0, 5.0)
        input_data['LBDLDLSI'] = st.slider("LDL Cholesterol", 1.0, 8.0, 3.0)
        input_data['LBDTRSI'] = st.slider("Triglycerides", 0.5, 10.0, 1.5)
        
    with col5:
        input_data['LBDAPBSI'] = st.slider("Apolipoprotein B", 0.1, 2.0, 1.0)
        input_data['LBXHSCRP'] = st.slider("C-reactive Protein", 0.1, 10.0, 1.0)
        input_data['LBDINSI'] = st.slider("Insulin", 10.0, 300.0, 60.0)
        input_data['eLDL_Trig'] = st.slider("LDL/Trig Ratio", 0.1, 5.0, 0.5)

# Additional measurements
with st.expander("Other Measurements"):
    input_data['Fasting_hrs'] = st.slider("Fasting Hours", 0.0, 24.0, 12.0)
    input_data['DXXSATA'] = st.slider("SAT Area", 100.0, 1000.0, 300.0)
    input_data['DXXSATM'] = st.slider("SAT Mass", 1000.0, 10000.0, 3000.0)
    input_data['DXXVFATA'] = st.slider("Visceral Fat Area", 50.0, 500.0, 100.0)
    input_data['DXXVFATM'] = st.slider("Visceral Fat Mass", 500.0, 5000.0, 1000.0)

# Prediction
if st.button("Predict Metabolic Syndrome", type="primary"):
    try:
        # Prepare input in correct format for multi-input model
        prediction_dict = {}
        for feature in feature_names:
            prediction_dict[feature] = np.array([[input_data[feature]]])
        
        # Make prediction
        prediction = model.predict(prediction_dict)
        probability = tf.sigmoid(prediction[0][0]).numpy()
        
        # Display results
        st.header("🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if probability > 0.5:
                st.error(f"🚨 HIGH RISK")
                st.metric("Metabolic Syndrome", "LIKELY", delta=f"{probability:.1%}")
            else:
                st.success(f"✅ LOW RISK") 
                st.metric("Metabolic Syndrome", "UNLIKELY", delta=f"{probability:.1%}")
        
        with col2:
            st.metric("Probability", f"{probability:.1%}")
            st.progress(float(probability))
        
        # Risk interpretation
        st.subheader("Risk Interpretation")
        if probability < 0.3:
            st.info("🟢 Low risk - Healthy metabolic profile")
        elif probability < 0.7:
            st.warning("🟡 Moderate risk - Consider lifestyle changes")
        else:
            st.error("🔴 High risk - Medical consultation recommended")
            
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Metabolic Syndrome Criteria:**
    - Abdominal obesity
    - Elevated triglycerides  
    - Low HDL cholesterol
    - High blood pressure
    - Elevated fasting glucose
    """)
    
    st.header("Model Info")
    st.write(f"• Input features: {len(feature_names)}")
    st.write("• Model format: .keras")
    st.write("• Architecture: Neural Network")