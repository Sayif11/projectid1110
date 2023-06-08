# importing th libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
import itertools
import nltk

nltk.download('stopwords')
# reading the dataset
news_dataset = pd.read_csv('/home/nalan/Documents/data_sets/train.csv')
news_dataset.head()

# checking the sum of blank values in the data and filling with empty string
news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')

# concatansting the title and author
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
port_stem = PorterStemmer()


# data preprocessing
def remove_non_alphabetic(content):
    return re.sub('[^a-zA-Z]', '', content)


def convert_to_lowercase(content):
    return content.lower()


def preprocess(content):
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


# applying the stemmming function
news_dataset['content'] = news_dataset['content'].apply(preprocess)


# separating the content and the label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# converting to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# splittng the data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=1
)
# model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy*100)


# function for plotting confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# show the matrix
cm = metrics.confusion_matrix(Y_test, X_test_prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
plt.show()
