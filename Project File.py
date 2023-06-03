import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

news_dataset = pd.read_csv(r"C:/Users/rohit/Downloads/train.csv")

news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')

news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
print(X)
import nltk
nltk.download('stopwords')

port_stem = PorterStemmer()

def preprocess(content):
    content = re.sub('[^a-zA-Z]',' ',content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content

news_dataset['content'] = news_dataset['content'].apply(preprocess)

X = news_dataset['content'].values
Y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, stratify=Y, random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
import matplotlib.pyplot as plt
import seaborn as sns


rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)

rf_model = rf.fit(X_train, Y_train)

y_pred = rf_model.predict(X_test)

precision, recall, fscore, train_support = score(Y_test, y_pred, pos_label=1, average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(Y_test,y_pred), 3)))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
class_label = [0, 1]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()