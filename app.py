import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Predictor")
st.markdown("Enter the car details to get the estimated price.")

BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, 'car_price_predictor_model.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'Cleaned_Car_data.csv')

@st.cache_resource
def load_data():
    try:
        model = pickle.load(open(MODEL_PATH, 'rb'))
        car = pd.read_csv(CSV_PATH)
        return model, car
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, car = load_data()

if model is not None and car is not None:
    st.sidebar.header("Car Details")
    
    company = st.selectbox("Select Company", sorted(car['company'].unique()))
    
    car_models = sorted(car[car['company'] == company]['name'].unique())
    car_model = st.selectbox("Select Model", car_models)
    
    year = st.selectbox("Select Year", sorted(car['year'].unique(), reverse=True))
    
    fuel_type = st.selectbox("Fuel Type", car['fuel_type'].unique())
    

    driven = st.slider("Kilometers Driven", min_value=0, max_value=400000, step=1000, value=10000)

    if st.button("Predict Price"):
        try:
            input_df = pd.DataFrame(
                [[car_model, company, year, driven, fuel_type]],
                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
            )
            
            prediction = model.predict(input_df)
            
            st.success(f"Estimated Price: TK {np.round(prediction[0], 2)}")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
else:
    st.warning("Required files not found. Please check your folder.")

# cd "C:\Users\USER\OneDrive\Desktop\Code practice\Machine_Learning_Projects\Car Price Predictor Project"
# flask --app app.py run
# streamlit run app.py