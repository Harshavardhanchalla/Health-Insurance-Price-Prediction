import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

with open ('insurance_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('HEALTH INSURENCE PRICE DATA.csv')  # Update with your dataset filename
    return df

# Function to preprocess the data
def preprocess_data(df):

    #df.drop(columns=['ID'], inplace=True)   # Remove ID column
    
    # Convert categorical variables to numerical
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
    df['smoker'] = df['smoker'].map({'No': 0, 'Yes': 1})
    df['Diabetes'] = df['Diabetes'].map({'No': 0, 'Yes': 1})
    df['Blood Pressure'] = df['Blood Pressure'].map({'No': 0, 'Yes': 1})

    # Handle missing values (if any)
    df.fillna(0, inplace=True)  # Replace missing values with 0 for simplicity
    
    return df

# Train the model
def train_model(X_train, y_train):
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model

# Function to predict
def predict(model, X_test):
    return model.predict(X_test)

# Load the data
df = load_data()

# Preprocess the data
df = preprocess_data(df)

# Split into features and target variable
X = df.drop(columns=['Insurance Price'])
y = df['Insurance Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)

# Streamlit UI
st.title('Health Insurance Price Prediction')

st.sidebar.header('User Input Features')

# Collect user inputs
def user_input_features():
    Age = st.sidebar.text_input('Age', '0')
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    Diabetes = st.sidebar.selectbox('Diabetes', ('Yes', 'No'))
    BloodPressure = st.sidebar.selectbox('Blood Pressure', ('Yes', 'No'))
    Height = st.sidebar.text_input('Height', '0')
    Weight = st.sidebar.text_input('Weight', '0')
    bmi = st.sidebar.text_input('BMI', '0')
    children = st.sidebar.text_input('Number of Children', '0')
    smoker = st.sidebar.selectbox('Smoker', ('No', 'Yes'))
    user_data = {'Age': int(Age), 'sex': sex, 'Diabetes':Diabetes, 'Blood Pressure':BloodPressure, 'Height':int(Height), 'Weight':int(Weight), 'bmi': float(bmi), 'children': int(children), 'smoker': smoker}
    features = pd.DataFrame(user_data, index=[0])
    return features

# Predict insurance charges
def main():

    # Collecting user input features
    input_df = user_input_features()
    
    # Convert categorical variables to numerical
    input_df['sex'] = input_df['sex'].map({'Male': 0, 'Female': 1})
    input_df['smoker'] = input_df['smoker'].map({'No': 0, 'Yes': 1})
    input_df['Diabetes'] = input_df['Diabetes'].map({'No': 0, 'Yes': 1})
    input_df['Blood Pressure'] = input_df['Blood Pressure'].map({'No': 0, 'Yes': 1})

    # Submit button
    submit = st.sidebar.button('Predict')
    if submit:
        # Assuming you have a function `predict()` which takes the DataFrame and returns a prediction
        # Make sure to handle the prediction part here as per your model requirements
        prediction = model.predict(input_df)  # Assuming `model` is preloaded and can handle this DataFrame
        
        # Output prediction
        # Assuming the prediction is returned as a single value or as an array
        prediction = np.array(prediction).flatten()[0]  # Adjust according to your model output
        st.write(f"Estimated Insurance Cost: Rs.{round(prediction, 2)}")

if __name__ == '__main__':
    main()
