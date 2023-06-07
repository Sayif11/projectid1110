import numpy as np
import pandas as pd
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Download required NLTK modules

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

np.random.seed(42)

# set CPU cores

num_cores = os.cpu_count()

# inserting the data

collection = pd.read_csv('/home/nalan/Documents/data_sets/train.csv')

# Preprocess the data by parallel processing


def preprocess_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    final_words = []
    for word, tag in nltk.pos_tag(tokens):
        if word not in stopwords.words('english') and word.isalpha():
            word_final = lemmatizer.lemmatize(word, tag_map[tag[0]])
            final_words.append(word_final)
    return ' '.join(final_words)


collection['text_final'] = collection['text'].apply(preprocess_text)

# Prepare for training and testing

x = collection['text_final']
y = collection['label']
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=1)

# Word vectorization 

word_vector = TfidfVectorizer(max_features=5000)
Train_X_word = word_vector.fit_transform(Train_X)
Test_X_word = word_vector.transform(Test_X)

# SVM classifier

SVM = svm.SVC(C=1.0, kernel='linear', gamma='auto')

# Training the model 
SVM.fit(Train_X_word, Train_Y, sample_weight=None)

# Prediction using parallel processing
SVM_prediction = SVM.predict(Test_X_word)

# Calculating accuracy score

accuracy = metrics.accuracy_score(SVM_prediction, Test_Y)
print("SVM Accuracy Score ->", accuracy * 100)

# plotting the confusion matrix

cm = metrics.confusion_matrix(Test_Y, SVM_prediction)
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[False, True]
    )
cm_display.plot()
plt.show()




