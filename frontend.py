import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re

nltk.download('stopwords')
# Load the trained model
port_stem = PorterStemmer()
vectorizer = TfidfVectorizer()
def preprocess(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content
def predict_label(text):
    preprocessed_text = preprocess(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Create the Streamlit app
def main():
    st.title("Fake News Detection")
    news_dataset= []
    # Input text box
    title = st.text_input("Enter the news text")
    content = st.text_input("enter the content")
    author = st.text_input("emter the author name")

    if st.button("Predict"):
if __name__ == '__main__':
    main()


