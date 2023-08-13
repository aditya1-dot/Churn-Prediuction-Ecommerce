import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
best_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Load the cleaned dataset
data = pd.read_csv('cleaned_dataset.csv')

# Separate features and target
X = data.drop(columns=['CustomerID', 'Churn'])
y = data['Churn']

# Fit a StandardScaler on the entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model on the entire dataset
best_model.fit(X_scaled, y)

def predict_churn(user_input):
    # Preprocess the user input
    user_input_scaled = scaler.transform(user_input)

    # Predict churn for the user input
    churn_prediction = best_model.predict(user_input_scaled)

    return churn_prediction[0]

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Enter user details to predict churn status:")

# Get user input
user_input = []

# Get the list of feature columns
feature_columns = data.columns.tolist()
for feature in feature_columns:
    if feature in ['CustomerID', 'Churn']:
        continue
    user_value = st.number_input(f"Enter value for '{feature}':", key=feature)
    user_input.append(user_value)

if st.button("Predict Churn"):
    # Create a DataFrame from the user input
    user_input_df = pd.DataFrame([user_input], columns=feature_columns[2:])  # Exclude CustomerID and Churn

    # Preprocess the user input and predict churn
    churn_result = predict_churn(user_input_df.values)

    # Display prediction result
    if churn_result == 0:
        st.write("Prediction: Not Churn")
    else:
        st.write("Prediction: Churn")
