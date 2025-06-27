import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_data, preprocess_data
import plotly.express as px

# Load model
model_path = os.path.join("..", "models", "model.pkl")
model = joblib.load(model_path)

# Feature names (in order)
feature_names = [
    'AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
    'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
    'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
    'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
]

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Lung Cancer Risk Estimator", "Model Explainability"])

# Utility: Get user input
def get_user_input():
    st.sidebar.markdown("### Enter Patient Information")
    age = st.sidebar.slider("Age", 20, 100, 40)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    smoking = st.sidebar.selectbox("Do you smoke?", ("Yes", "No"))
    yellow_fingers = st.sidebar.selectbox("Yellow fingers?", ("Yes", "No"))
    anxiety = st.sidebar.selectbox("Do you feel anxious?", ("Yes", "No"))
    peer_pressure = st.sidebar.selectbox("Peer Pressure?", ("Yes", "No"))
    chronic = st.sidebar.selectbox("Chronic disease?", ("Yes", "No"))
    fatigue = st.sidebar.selectbox("Fatigue?", ("Yes", "No"))
    allergy = st.sidebar.selectbox("Allergy?", ("Yes", "No"))
    wheezing = st.sidebar.selectbox("Wheezing?", ("Yes", "No"))
    alcohol = st.sidebar.selectbox("Alcohol consumption?", ("Yes", "No"))
    coughing = st.sidebar.selectbox("Coughing?", ("Yes", "No"))
    sob = st.sidebar.selectbox("Shortness of breath?", ("Yes", "No"))
    swallowing = st.sidebar.selectbox("Swallowing difficulty?", ("Yes", "No"))
    chest_pain = st.sidebar.selectbox("Chest pain?", ("Yes", "No"))

    binary_map = {"Yes": 1, "No": 0}
    gender_val = 1 if gender == "Male" else 0

    features = np.array([[
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
        binary_map[swallowing],
        binary_map[chest_pain]
    ]])

    return features

# 🧪 Page 1: Lung Cancer Risk Prediction
if page == "Lung Cancer Risk Estimator":
    st.title("🫁 Lung Cancer Risk Estimator")

    user_features = get_user_input()

    if st.button("Predict Lung Cancer Risk"):
        result = model.predict(user_features)[0]
        if result == 1:
            st.warning("⚠️ High Risk of Lung Cancer")
        else:
            st.success("✅ Low Risk of Lung Cancer")

# 📊 Page 2: Model Explainability Dashboard
elif page == "Model Explainability":
    st.title("📊 Model Explainability Dashboard (Enhanced with Plotly)")

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    # ✅ 1. Overall Accuracy
    st.markdown("### ✅ Overall Model Accuracy")
    st.success(f"The model achieved an accuracy of **{accuracy * 100:.2f}%** on the test set.")

    # 🥧 2. Class Distribution Pie Chart
    st.markdown("### 🥧 Class Distribution in Dataset")
    class_counts = df['LUNG_CANCER'].value_counts().rename({1: 'YES', 0: 'NO'}).reset_index()
    class_counts.columns = ['LUNG_CANCER', 'Count']
    pie_chart = px.pie(class_counts, names='LUNG_CANCER', values='Count', title='Lung Cancer Class Distribution')
    st.plotly_chart(pie_chart)

    # 📊 3. Feature Importance
    st.markdown("### 📊 Feature Importance (Random Forest)")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names_df = df.drop('LUNG_CANCER', axis=1).columns
        importance_df = pd.DataFrame({
            'Feature': feature_names_df,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        bar_chart = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                           title="Feature Importance", color='Importance',
                           color_continuous_scale='blues')
        st.plotly_chart(bar_chart)

    # 📋 4. Classification Report
    st.markdown("### 📋 Classification Report")
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    # 🔁 5. Confusion Matrix
    st.markdown("### 🔁 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual: NO', 'Actual: YES'], columns=['Predicted: NO', 'Predicted: YES'])
    heatmap = px.imshow(cm_df, text_auto=True, color_continuous_scale='blues', title="Confusion Matrix")
    st.plotly_chart(heatmap)

