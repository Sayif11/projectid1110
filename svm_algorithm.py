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
collection['text']=collection['text'].str.lower()
collection['text']=collection['text'].astype(str).apply(lambda x: word_tokenize(x) )
print(collection['text'])

#Geting punkt and wordnet
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# removing blank, converting to lower case
# and  tokenizing
collection['text'].dropna(inplace=True)
collection['text']=[str(i).lower for i in collection['text']]
collection['text']= [word_tokenize(str(i)) for i in collection['text']]
# creating default dictionary and assigning keys
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(collection['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    collection.loc[index,'text_final'] = str(Final_words)

# preparing for train and test
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    collection['text_final'],collection['label'],test_size=0.2,random_state=1)

# word vectorization
word_vector=TfidfVectorizer(max_features=5000)
word_vector.fit(collection['text_final'])
Train_X_word=word_vector.transform(Train_X)
Test_X_word=word_vector.transform(Test_X)
print(word_vector.vocabulary_)
print(Train_X_word)

# svm classifier
SVM=svm.SVC(C=1.0,kernel='linear',gamma='auto')
SVM.fit(Train_X_word,Train_Y)
SVM_prediction=SVM.predict(Test_X_word)
accuracy= accuracy_score(SVM_prediction, Test_Y)
print("SVM Accuracy Score -> ",accuracy*100)
cm=metrics.confusion_matrix(Test_Y,SVM_prediction)
cm_display=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[False,True])
cm_display.plot()
plt.show()

#Preparing the test set for prediction
collection_test= pd.read_csv('/home/nalan/Documents/data_sets/test.csv')

#Geting punkt and wordnet
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
collection_test['text'].dropna(inplace=True)
collection_test['text'] = [str(entry).lower() for entry in collection_test['text']]
collection_test['text']= [word_tokenize(str(entry)) for entry in collection_test['text']]
for index,entry in enumerate(collection_test['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    collection_test.loc[index,'text_final'] = str(Final_words)
    
# word vectorization
Test_X=collection_test['text_final']
word_vector_test=TfidfVectorizer(max_features=5000)
word_vector_test.fit(Test_X)
Test_X=word_vector.transform(Test_X)
print(word_vector_test.vocabulary_)

lable_prediction=SVM.predict(Test_X)
result=pd.DataFrame({'id': collection_test['id'],'label':lable_prediction})
result.to_csv('submit.csv',index=False)



