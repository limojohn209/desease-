import streamlit as st
import pandas as pd
import joblib

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Diabetes Risk Prediction App")
st.write("Jibu maswali yafuatayo ili kupata makadirio ya hatari ya kisukari. (Hii si uchunguzi wa kitabibu - wasiliana na daktari kwa vipimo halisi.)")

age = st.number_input("Age", min_value=15, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
waist = st.number_input("Waist Circumference (cm)", min_value=50.0, max_value=170.0, value=85.0)

family_history = st.selectbox("Family History of Diabetes", ["Yes", "No"])
physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
blood_pressure = st.selectbox("Blood Pressure", ["Normal", "Elevated", "High"])
smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
mental_stress = st.selectbox("Mental Health / Stress Level", ["Low", "Moderate", "High"])
ethnicity = st.selectbox("Ethnicity", ["African", "Asian", "European", "Other"])
social_environment = st.selectbox("Social Environment", ["Urban", "Rural"])
fruit_veg = st.selectbox("Fruit & Vegetable Intake", ["Low", "Moderate", "High"])
sugar_drinks = st.selectbox("Sugary Drinks Frequency", ["Rarely", "Sometimes", "Often"])
processed_food = st.selectbox("Processed Food Frequency", ["Rarely", "Sometimes", "Often"])
alcohol = st.selectbox("Alcohol Consumption", ["Never", "Moderate", "Heavy"])
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=2.0, max_value=12.0, value=7.0)
sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good"])

if st.button("Predict"):
    input_dict = {
        "Age": age,
        "BMI": bmi,
        "Waist_Circumference": waist,
        "Family_History": family_history,
        "Physical_Activity": physical_activity,
        "Blood_Pressure": blood_pressure,
        "Smoking": smoking,
        "Mental_Health_Stress": mental_stress,
        "Ethnicity": ethnicity,
        "Social_Environment": social_environment,
        "Fruit_Veg_Intake": fruit_veg,
        "Sugar_Drinks_Frequency": sugar_drinks,
        "Processed_Food_Frequency": processed_food,
        "Alcohol_Consumption": alcohol,
        "Sleep_Duration_Hours": sleep_duration,
        "Sleep_Quality": sleep_quality
    }

    sample = pd.DataFrame([input_dict])

    categorical_cols = ["Family_History","Physical_Activity","Blood_Pressure","Smoking",
                         "Mental_Health_Stress","Ethnicity","Social_Environment",
                         "Fruit_Veg_Intake","Sugar_Drinks_Frequency",
                         "Processed_Food_Frequency","Alcohol_Consumption","Sleep_Quality"]

    for col in categorical_cols:
        le = label_encoders[col]
        sample[col] = le.transform(sample[col])

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    probability = model.predict_proba(sample_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"Matokeo: Uwezekano wa hatari ya Kisukari (Probability: {probability:.2%})")
    else:
        st.success(f"Matokeo: Hatari ndogo ya Kisukari (Probability: {probability:.2%})")

    st.caption("Kumbuka: Hii ni zana ya makadirio tu, sio uchunguzi wa kitabibu.")
