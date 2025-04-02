import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Models & Vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("naive_bayes_model.pkl", "rb") as f:
    naive_bayes_model = pickle.load(f)

with open("decision_tree_model.pkl", "rb") as f:
    decision_tree_model = pickle.load(f)

# Function to Predict Spam
def predict_email(email_text, model_choice):
    cleaned_text = [" ".join(email_text.lower().split())]  # Clean input text
    vectorized_text = vectorizer.transform(cleaned_text)

    if model_choice == "Logistic Regression":
        prediction = logistic_model.predict(vectorized_text)
    elif model_choice == "Naïve Bayes":
        prediction = naive_bayes_model.predict(vectorized_text)
    else:
        prediction = decision_tree_model.predict(vectorized_text)

    return "Not Spam" if prediction[0] == 0 else "Spam"

# Streamlit UI
st.title("Email Spam Detector")

st.sidebar.header("Choose Classifier")
model_choice = st.sidebar.selectbox("Select a Model", ["Logistic Regression", "Naïve Bayes", "Decision Tree"])

email_text = st.text_area("Paste Email Text Here:")

if st.button("Predict"):
    result = predict_email(email_text, model_choice)
    st.write("### Prediction:", result)
