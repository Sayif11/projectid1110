# importing the needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# downloading stopwords from nltk
nltk.download('stopwords')

# reading the dataset to pandas
news_dataset = pd.read_csv('/home/nalan/Documents/data_sets/train.csv')
news_dataset.head()

# checking the dimension of the data set
news_dataset.shape

# finding the sum of empty value aand filling them with empty string
news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')

# concatanating the author and title part
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

port_stem = PorterStemmer()


# function for preproccesing
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


# applying the preprocess function to the dataset
news_dataset['content'] = news_dataset['content'].apply(preprocess)

# separating the content and the label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# change string to float
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


# show the confusion matrix
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
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# preparing train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)


# word vectorization
pipe = Pipeline([('model', DecisionTreeClassifier(criterion='entropy',
                                                  max_depth=20,
                                                  splitter='best',
                                                  random_state=42))])

# model training
model = pipe.fit(X_train, Y_train)

# test prediction
prediction = model.predict(X_test)
print("accuracy: ", accuracy_score(Y_test, prediction) * 100)

# show the confusion matrix

cm = metrics.confusion_matrix(Y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
plt.show()
