# Programming Course project #
## Fake news detection ##

Social Media has emerged as an essential medium of communication. As social media platforms have become popular, more people prefer to search for news in social media instead of conventional news sources. Although consuming news on social media offers many benefits, the absence of control and convenient access has led to the wide dissemination of fake news or misinformation.The main goal of this system is to develop a model that can distinguish between genuine (“real”) and fabricated (“fake”) news. This task is a classic text classification problem that accurately classifies news articles based on their credibility. The model is trained with 80% of the dataset and the rest is used for testing.  

## Dataset ## 
Data: https://www.kaggle.com/c/fake-news/data?select=train.csv  
  description:
  A full training dataset with the following attributes:  
  id: unique id for a news article  
  title: the title of a news article  
  author: author of the news article  
  text: the text of the article; could be incomplete  
  label: a label that marks the article as potentially unreliable  
         1: unreliable  
         0: reliable  
## Files ##

1. Model_decision.py : code for the model using decision tree classifier algorithm  
2. model_logistic.py : code for the model using logistic regression algorithm  
3. svm_algorithm.py : code for the model using support vector machine algorithm  
4. Random_forest.py : code for the model using random tree classifier algorithm  
5. frontend_for_the_RF.py : code fo the streamlit web app using RF classifier  
6. trained_model.joblib : saved file containing trained model using the Random forest algorithm  
7. vectorizer.joblib : saved file containing trained TF-IDF class  
8. requirements.txt : the required modules for running of the program

## basic methadology ##
1.	Dataset: The Fake News dataset in kaggle ( Fake News | Kaggle) has been used to train this model. The data is divided 80%for training and 20% for testing  after the construction of the model. The title and author part is taken for training for simplicity's sake.
2.	preprocessing : The input text is first converted to lower case and removed punctuations, stop words, and nonalphanumeric characters. Natural Language Toolkit (NLTK) library of python is used for these preprocessing tasks.
3.	Extraction: We used the scikit-learn function TfidfVectorizer for converting the text data to a matrix of TFIDF features
4.	Training :
a, Logistic regression: The Logistic Regression model is trained with
the training data set. The scikit-learn implementation of Logistic Regression is used.
	b, Decision Tree classifier: The scikit-learn implementation of decision tree is used with parameters criterion=entropy
	c, Random forest classifier: Random Forest Classifier: The scikit-learn implementation of random forest is used with parameters
n estimators=150, criterion=‘gini
	D, Support vector Machine: The scikit-learn implementation of SVM is used with parameter kernel=‘linear’. The model is trained with the training data set
5.	Testing: The model is tested with 20% of the data
6.	 Accuracy Score: accuracy score is found and using the Sklearn module the confusion matrixes are plotted  
7.	The most accurate model is chosen and is used to create a web app for the detection of fake news using the headlines and autor name.


## Future work ##
- The frontend have significant bugs which are yet to be fixed, which is a problem we are facing and would like to work in the future
- Adding the text part of the dataset to the training will increase the accuracy and give more meaningfull results 
- using a dataset of tweets we can modify the odel to track fake news spreading on Twitter
Group participants:
Mouhamad Sayif,
Nalan S,
Siriki Sai Tejeshwar.
