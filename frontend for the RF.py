# importing libraries
import streamlit as st
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Load the trained model
loaded_model = joblib.load(
    r'C:\Users\hp\Documents\GitHub\projectid1110\trained_model.joblib'
)


# Load the TfidfVectorizer
vectorizer = joblib.load(
    r'C:\Users\hp\Documents\GitHub\projectid1110\vectorizer.joblib'
)


# Function to preprocess the input text
def preprocess(content):
    port_stem = PorterStemmer()
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [
        port_stem.stem(word)
        for word in content
        if word not in stopwords.words('english')
    ]
    content = ' '.join(content)
    return content


# Function to predict
def predict(news):
    preprocessed_news = preprocess(news)
    if not preprocessed_news:
        return None
    X = vectorizer.transform([preprocessed_news])
    prediction = loaded_model.predict(X)
    return prediction[0]


# creating streamlit web app
def main():
    st.title("Fake News Classifier")
    # User input
    title = st.text_area("Enter the news title", height=6)
    author = st.text_area("Enter the author's name")
    user_input = title + '' + author
    # Classify button
    if st.button("predict"):
        if user_input:
            prediction = predict(user_input)
            if prediction == 0:
                st.error("This news is classified as **fake**.")
            else:
                st.success("This news is classified as **real**.")
        else:
            st.warning("Please enter some news text.")


# Run the app
if __name__ == "__main__":
    main()



