import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved models and scaler
#model = pickle.load(open('svm_model.sav', 'rb'))  # main prediction model (SVM)

# Model selector
#model_choice = st.selectbox(
    #"🧠 Choose a Model for Prediction",
  #["🧩 SVM", "🌲 Random Forest", "📊 Logistic Regression", "👥 KNN"]
#)
model_choice = st.selectbox(
    "🧠 Choose Prediction Method (Model)",
    [
        "💻 Basic (Support Vector Machine - SVM)",
        "🌲 Advanced (Random Forest)",
        "📊 Classic (Logistic Regression)",
        "👥 Nearest Match (K-Nearest Neighbors - KNN)"
    ],
    help="Different methods used by the app to predict diabetes. You can try each one!"
)
st.markdown("### 👤 Personal & Health Details (Please fill accurately)")

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
  Pregnancies = st.number_input('Pregnancies (count)', min_value=0,
  step=1,help="Enter how many times the person has been pregnant (0 if never)")
  Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0,help="Enter fasting blood sugar level. Normal is around 70–100 mg/dL")
  BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0,help="Diastolic blood pressure (bottom number). Normal is around 80 mm Hg")
  SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0,help="Skin thickness near tricep muscle. Measured using calipers")


with col2:
  Insulin = st.number_input('Insulin (mu U/mL)', min_value=0,help="2-hour serum insulin level. Normal range is around 16–166 mu U/mL")
  BMI = st.number_input('BMI (kg/m²)', min_value=0.0, help="BMI = weight/height². Normal: 18.5 to 24.9")
  DPF = st.number_input('Diabetes Pedigree Function (ratio)', min_value=0.0,help="Likelihood of diabetes based on family history (0 to 2+)")
  Age = st.number_input('Age(years)', min_value=1, help="Enter the person’s age in years")

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

