import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved models and scaler
#model = pickle.load(open('svm_model.sav', 'rb'))  # main prediction model (SVM)

# Model selector
model_choice = st.selectbox(
    "🧠 Choose a Model for Prediction",
  ["🧩 SVM", "🌲 Random Forest", "📊 Logistic Regression", "👥 KNN"]
)

# Load the selected model
if "SVM" in model_choice:
    model = pickle.load(open("svm_model.sav", "rb"))
elif "Random Forest"in model_choice:
    model = pickle.load(open("rf_model.sav", "rb"))
elif  "Logistic Regression" in model_choice:
    model = pickle.load(open("logreg_model.sav", "rb"))
else:
    model = pickle.load(open("knn_model.sav", "rb"))

scaler = pickle.load(open('scaler.sav', 'rb'))    # feature scaler
rf_model = pickle.load(open('rf_model.sav', 'rb'))  # random forest for feature importance

# Define feature names (same order as dataset)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

st.title("🩺 Diabetes Prediction App")
st.subheader("📥 Enter Patient Data")
#st.markdown("Enter patient health details to predict diabetes risk:")

# Input fields
col1, col2 = st.columns(2)
with col1:
  Pregnancies = st.number_input('Pregnancies', min_value=0, step=1)
  Glucose = st.number_input('Glucose Level', min_value=0)
  BloodPressure = st.number_input('Blood Pressure', min_value=0)
  SkinThickness = st.number_input('Skin Thickness', min_value=0)


with col2:
  Insulin = st.number_input('Insulin', min_value=0)
  BMI = st.number_input('BMI', min_value=0.0)
  DPF = st.number_input('Diabetes Pedigree Function', min_value=0.0)
  Age = st.number_input('Age', min_value=1)

# Prediction logic
if st.button('Predict Diabetes'):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DPF, Age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("🧬 The person is likely to have diabetes.")
    else:
        st.success("✅ The person is unlikely to have diabetes.")

# Feature importance section
st.subheader("📊 Feature Importance (from Random Forest)")

importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)



# Plot using seaborn

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df,
    palette='Blues_r',
    hue='Feature',
    dodge=False,
    legend=False,
    ax=ax
)
ax.set_title('Feature Importance (Random Forest)')
st.pyplot(fig)

