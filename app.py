import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Email Detector")

# Input box
message = st.text_area("Enter your email text:")

if st.button("Predict"):

    # Convert text to vector
    message_vec = vectorizer.transform([message])

    # Predict
    prediction = model.predict(message_vec)

    # Output
    if prediction[0] == 1:
        st.error("🚨 Spam")
    else:
        st.success("✅ Not Spam")