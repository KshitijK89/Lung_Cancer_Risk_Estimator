import streamlit as st
import joblib
import numpy as np
import os

# Load the model
model_path = os.path.join("..", "models", "model.pkl")
model = joblib.load(model_path)

# Page title
st.title("🫁 Lung Cancer Risk Estimator")

# Sidebar input form
st.sidebar.header("Enter Patient Information")

# User input fields
def user_input():
    age = st.sidebar.slider("Age", 20, 100, 40)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    smoking = st.sidebar.selectbox("Do you smoke?", ("Yes", "No"))
    yellow_fingers = st.sidebar.selectbox("Yellow fingers?", ("Yes", "No"))
    anxiety = st.sidebar.selectbox("Do you feel anxious?", ("Yes", "No"))
    chronic = st.sidebar.selectbox("Chronic disease?", ("Yes", "No"))
    fatigue = st.sidebar.selectbox("Fatigue?", ("Yes", "No"))
    allergy = st.sidebar.selectbox("Allergy?", ("Yes", "No"))
    wheezing = st.sidebar.selectbox("Wheezing?", ("Yes", "No"))
    alcohol = st.sidebar.selectbox("Alcohol consumption?", ("Yes", "No"))
    coughing = st.sidebar.selectbox("Coughing?", ("Yes", "No"))
    sob = st.sidebar.selectbox("Shortness of breath?", ("Yes", "No"))
    peer_pressure = st.sidebar.selectbox("Peer Pressure?", ("Yes", "No"))
    swallowing_difficulty = st.sidebar.selectbox("Swallowing Difficulty?", ("Yes", "No"))
    chest_pain = st.sidebar.selectbox("Chest Pain?", ("Yes", "No"))


    # Encode manually: Yes = 1, No = 0
    binary_map = {"Yes": 1, "No": 0}
    gender_val = 1 if gender == "Male" else 0

    data = np.array([[
    age, gender_val,
    binary_map[smoking],
    binary_map[yellow_fingers],
    binary_map[anxiety],
    binary_map[peer_pressure],
    binary_map[chronic],
    binary_map[fatigue],
    binary_map[allergy],
    binary_map[wheezing],
    binary_map[alcohol],
    binary_map[coughing],
    binary_map[sob],
    binary_map[swallowing_difficulty],
    binary_map[chest_pain]
]])

    return data

# Collect input
features = user_input()

# Predict
if st.button("Predict Lung Cancer Risk"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.warning("⚠️ High Risk of Lung Cancer")
    else:
        st.success("✅ Low Risk of Lung Cancer")
