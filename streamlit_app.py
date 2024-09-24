import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.write("<h2 align='center'>Development of Machine Learning Driven Prediction of Tropical Agricultural Produce in Nigeria (Food Crop)</h2>", unsafe_allow_html=True)

st.image("cover.jpg")

st.write("""Image Credit: <a href='https://saroagrosciences.com/2023/06/04/5-food-crops-that-guarantee-high-financial-returns-in-nigeria/'>Saro Agro Sciences</a>""", unsafe_allow_html=True)

st.write("""
## About

<p align="justify">
Agriculture is a vital sector in Nigeria, contributing significantly to the nation's GDP and providing food security for millions. This project focuses on developing a machine learning-driven system to predict crop yields for key tropical agricultural produce such as maize, rice, cassava, and groundnut, among others. The prediction is based on critical factors like land area under cultivation, annual rainfall, fertilizer, and pesticide use. 
<br />
The goal is to assist farmers, policymakers, and other stakeholders in optimizing crop production by making informed decisions based on data. The machine learning model uses various input features to predict crop yield, offering an innovative solution to the challenges of food production in Nigeria. This project code is available on <a href="https://github.com/Oyebamiji-Micheal/Crop-Yield-Prediction" target="_blank" style="text-decoration: None">Github</a>.
</p>
""", unsafe_allow_html=True)

st.write("""
##### **Group 10 Semester Project**
""")

predict_crop_yield = st.button("Predict Crop Yield")

st.sidebar.header("Input Features")

crop = st.sidebar.selectbox(
    "Crop: The name of the crop cultivated", ('Coconut', 'Cotton', 'Pepper', 'Maize', 'Onion', 'Rice', 'Millet',
       'Sugarcane', 'Yam', 'Cassava', 'Ginger', 'Groundnut', 'Sorghum',
       'Cashew', 'Banana', 'Soybean', 'Cowpea')
)

area = st.sidebar.number_input(
    "Area: The total land area (in hectares) under cultivation for the specific crop.", min_value=1, max_value=100_000_000, value=1
)

annual_rainfall = st.sidebar.number_input(
    "Annual Rainfall: The annual rainfall received in the crop-growing region (in mm).", min_value=200.0, max_value=8_000.0, value=200.0
)

fertilizer = st.sidebar.number_input(
    "Fertilizer: The total amount of fertilizer used for the crop (in kilograms).", min_value=1.0, max_value=100_000_000.0, value=1.0
)

pesticide = st.sidebar.number_input(
    "Pesticide: The total amount of pesticide used for the crop (in kilograms).", min_value=0.01, max_value=10_000_000.0, value=0.01
)

if predict_crop_yield:
    data = {
        'Area': area,
        'Annual_Rainfall': annual_rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide,
        'Crop': crop
    }

    prediction_components = joblib.load('prediction_components.joblib')
    preprocessor = prediction_components['preprocessor']
    regressor_model = prediction_components['regressor_model']

    matrix = np.array(list(data.values())).reshape(1, -1)

    data = pd.DataFrame(matrix, columns=list(data.keys()))

    preprocessed_data = preprocessor.transform(data)

    prediction = regressor_model.predict(preprocessed_data)

    st.write(f"Predicted Crop Yield (production per unit area): {round(prediction[0], 2)}")
