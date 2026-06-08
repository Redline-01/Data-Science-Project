import streamlit as st
import pandas as pd
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Adult Income Prediction")
st.write(
    "Predict whether an individual's annual income exceeds $50,000."
)

# Sidebar

st.sidebar.header("Input Information")

age = st.sidebar.number_input("Age", 18, 90, 30)

fnlwgt = st.sidebar.number_input(
    "Final Weight",
    min_value=10000,
    max_value=1000000,
    value=100000
)

education_num = st.sidebar.slider(
    "Education Number",
    1,
    16,
    10
)

capital_gain = st.sidebar.number_input(
    "Capital Gain",
    0,
    100000,
    0
)

capital_loss = st.sidebar.number_input(
    "Capital Loss",
    0,
    5000,
    0
)

hours_per_week = st.sidebar.slider(
    "Hours per Week",
    1,
    100,
    40
)

workclass = st.sidebar.selectbox(
    "Workclass",
    [
        "?",
        "Private",
        "State-gov",
        "Federal-gov",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Local-gov",
        "Without-pay",
        "Never-worked"
    ]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    [
        "Widowed",
        "Divorced",
        "Separated",
        "Never-married",
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Married-AF-spouse"
    ]
)

occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "?",
        "Exec-managerial",
        "Machine-op-inspct",
        "Prof-specialty",
        "Other-service",
        "Adm-clerical",
        "Craft-repair",
        "Transport-moving",
        "Handlers-cleaners",
        "Sales",
        "Farming-fishing",
        "Tech-support",
        "Protective-serv",
        "Armed-Forces",
        "Priv-house-serv"
    ]
)

# Encode categories

workclass = encoders["workclass"].transform([workclass])[0]
marital_status = encoders["marital.status"].transform([marital_status])[0]
occupation = encoders["occupation"].transform([occupation])[0]

data = pd.DataFrame([[
    age,
    fnlwgt,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
    workclass,
    marital_status,
    occupation
]], columns=[
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "workclass",
    "marital.status",
    "occupation"
])

st.subheader("Entered Data")
st.dataframe(data)

# Prediction

if st.button("Predict Income"):

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("Income > $50K")
    else:
        st.error("Income <= $50K")



st.markdown("---")

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "85.57%")

with col2:
    st.metric("Precision", "71.83%")

with col3:
    st.metric("Recall", "62.89%")

with col4:
    st.metric("F1 Score", "67.06%")



st.markdown("---")

st.subheader("About Project")

st.write("""
This project uses the Adult Census Income dataset.

Algorithms evaluated:

- Logistic Regression
- Random Forest Classifier

Best Model:

- Random Forest
- Accuracy: 85.57%
- F1 Score: 67.06%

The model predicts whether an individual's annual income exceeds $50,000 based on demographic and employment information.
""")