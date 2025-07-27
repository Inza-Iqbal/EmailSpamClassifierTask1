import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')

# Load the trained model and vectorizer
model = pickle.load(open("trained_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing function
def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = text.lower().split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("📧 Email Spam Classifier")
st.write("Enter an email below to check if it's Spam or Not Spam:")

user_input = st.text_area("Email Text", height=200)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(user_input)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        result = "🛑 Spam" if prediction == 1 else "✅ Not Spam"
        st.subheader("Result:")
        st.success(result)
