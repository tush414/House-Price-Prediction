#!/usr/bin/env python
# coding: utf-8

# In[16]:


import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Load the cleaned data
data = pd.read_csv('Cleaned_data.csv')
def preprocess_input(user_location, user_total_sqft, user_bath, user_bhk):
        user_input = pd.DataFrame({'location': [user_location],'total_sqft': [user_total_sqft],'bath': [user_bath],'bhk': [user_bhk]
    })
        return user_input
# Separate features (X) and target variable (y)
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessor and XGBoost model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['total_sqft', 'bath', 'bhk']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['location'])
    ])
model = make_pipeline(preprocessor, XGBRegressor())
model.fit(X_train, y_train)

# Streamlit app
def main():
    st.title("House Price Prediction App")

    # User input form
    st.sidebar.header("User Input")
    user_location = st.sidebar.text_input("Enter the location:")
    user_bhk = st.sidebar.number_input("Enter the size:", value=1)
    user_total_sqft = st.sidebar.number_input("Enter the total sqft:", value=1000)
    user_bath = st.sidebar.number_input("Enter the number of bathrooms:", value=1)


    submitted = st.sidebar.button("Submit")

    # Make prediction when the user clicks the button
    if submitted:
        # Preprocess user input
        user_input = preprocess_input(user_location, user_total_sqft, user_bath, user_bhk)

        # Fit the entire pipeline (including ColumnTransformer)
        model.fit(X_train, y_train)  

        # Make predictions
        user_prediction = model.predict(user_input)

        # Display user input
        st.sidebar.subheader("User Input:")
        st.sidebar.write(user_input)

        # Display prediction
        st.subheader("Predicted Price:")
        st.success(f"Predicted Price: {user_prediction[0]:,.2f} L INR")

if __name__ == "__main__":
    main()


# In[ ]:




