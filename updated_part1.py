

import numpy as np 
import pandas as pd 
#Sklearn and nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score



import os
for dir,root,file in os.walk('/home/nalan/Documents/data_sets'):
    for filename in file:
        print(os.path.join(dir,filename))
np.random.seed(42)
collection= pd.read_csv('/home/nalan/Documents/data_sets/train.csv')
collection.head()
collection.tail()
collection.info()
collection.shape

x= collection.drop('label',axis=1)
x.head()
print(x)
y= collection['label']
print(y.head())

print(y.value_counts())
print(isinstance(collection['text'],(str)))
print(collection['text'])





