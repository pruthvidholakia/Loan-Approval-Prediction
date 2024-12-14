import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open("lgbm_top10_model.pkl", "rb"))

model = load_model()


# Define the input feature names
FEATURES = ['TotalAssets' ,'LoanAmount' ,'AnnualIncome' , 'MonthlyDebtPayments' ,'BaseInterestRate',
'LengthOfCreditHistory' ,'Age' , 'TotalLiabilities', 'CreditScore', 'CheckingAccountBalance']

# Features that require transformations
LOG_TRANSFORM_FEATURES = ['AnnualIncome', 'LoanAmount', 'TotalAssets', 'TotalLiabilities', 'CheckingAccountBalance']
SQRT_TRANSFORM_FEATURES = 'DebtToIncomeRatio'

# Custom CSS to set the background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://media.istockphoto.com/id/1661960136/photo/backlit-black-piggy-bank-on-dark-background-illustration-of-the-concept-of-financial.jpg?s=612x612&w=0&k=20&c=BsXndZdwicirdY50vjBigXLAXy0t4St0xMV0E-3JF9o=");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''

# Apply the custom CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit UI
st.title("Loan Approval Prediction")
st.write("""
This web app predicts if your loan will get approved or not based on your financial and credit history details.
""")

user_input = {}
for feature in FEATURES:
    if feature == 'BaseInterestRate' or feature == 'DebtToIncomeRatio':
        user_input[feature] = st.slider(f"Enter {feature} (0 to 1):", min_value=0.0, max_value=1.0, step=0.01)
    else:
        user_input[feature] = st.number_input(f"Enter {feature} (integer):", min_value=0, value=0, step=1)

# Predict button
if st.button("Predict Loan Approval"):
    # Convert user input to a NumPy array
    input_features = np.array([user_input[feature] for feature in FEATURES])

    # Apply transformations
    processed_features = input_features.copy()
    for i, feature in enumerate(FEATURES):
        if feature in LOG_TRANSFORM_FEATURES:
            processed_features[i] = np.log1p(processed_features[i])
        elif feature in SQRT_TRANSFORM_FEATURES:
            processed_features[i] = np.sqrt(processed_features[i])

    # Scale the features using MinMaxScaler
    processed_features = np.array(processed_features).reshape(1,-1)

    # Make prediction
    prediction = model.predict(processed_features)[0]
    prediction_proba = model.predict_proba(processed_features)[0][1]

    # Display the result
    if prediction == 1:
        st.success(f"Your loan is likely to be approved! Confidence: {prediction_proba:.2f}")
    else:
        st.error(f"Your loan is unlikely to be approved. Confidence: {1 - prediction_proba:.2f}")
