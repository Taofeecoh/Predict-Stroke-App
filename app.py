import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df = df.dropna(subset=['bmi'], axis=0)
    df.drop(['id', 'ever_married'], axis=1, inplace=True)
    df.loc[df['gender'] == 'Other', 'gender'] = 'Not sure'

    return df

df = load_data()

# Function to load model
def load_model():
    with open('model.pkl','rb') as file:
        datas = pickle.load(file)
    return datas

data = load_model()

rf_classifier_loaded = data["model"]
le_gender = data["le_gender"]
le_work = data["le_work"]
le_residence = data["le_residence"]
le_smokes = data["le_smokes"]

# Create page
st.title("Stroke Prediction App")
st.write("""### Information to predict whether an individual is likely to have stroke or not""")


# User input features
Gender = (
    "Male",
    "Female",
    "Not sure"
)
Work_type =(
    'Private', 
    'Self-employed', 
    'Govt_job', 
    'Children', 
    'Never_worked'
)

Residence = (
    "Rural",
    "Urban"
)
Smoking_Status = (
    'formerly smoked', 
    'never smoked', 
    'smokes', 
    'Unknown'
)

# User inputs
gender = st.selectbox("Gender", Gender)
work_type = st.selectbox("Work type", Work_type)
residence = st.selectbox("Residence", Residence)
smoking_status = st.selectbox("Please select smoking status", Smoking_Status)
hypertension_status = st.selectbox("Hypertensive? : Select 1 for YES, 0 for NO", [0,1])
heart_disease = st.selectbox("Heart disease? : Select 1 for YES, 0 for NO",[0,1])

# st.sidebar.header('User Input Parameters')
age = st.number_input("Age", placeholder="Input age...")
st.write(age)
bmi = st.slider("BMI", 0, 100, 3)
avg_glucose_level = st.slider("Average Glucose level", 0, 400, 20)

show_pred = st.button("Display result")
if show_pred:
    # X_test =  np.array([['Male', 55, 1, 1, 'Govt_job', 'Rural', 300, 300, 'smokes']])
    X_test = np.array([[gender, age, hypertension_status, heart_disease, work_type, residence, avg_glucose_level, bmi, smoking_status]])
    
    X_test[:,0] = le_gender.transform(X_test[:,0])
    X_test[:,4] = le_work.transform(X_test[:,4])
    X_test[:,5] = le_residence.transform(X_test[:,5])
    X_test[:,8] = le_smokes.transform(X_test[:,8])
    X_test = X_test.astype(float)
    #X_test

    # Make predictions
    pre = rf_classifier_loaded.predict(X_test)
    pre = pre[0]

    if pre == 0:
        st.subheader(f"No risk of developing Stroke")
    else:
        st.subheader(f"There is risk of developing Stroke")
