import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# Reading the news dataset from a CSV file
news_dataset = pd.read_csv('/home/nalan/Documents/data_sets/train.csv')

# Checking for missing values in the dataset and fill them with empty string
news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')

# For combining 'author' and 'title' columns to create the 'content' column
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

# To separate features (x) and labels(y)
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
print(X)
nltk.download('stopwords')

port_stem = PorterStemmer()

# Preprocessing the function to clean the text content


def stemmming(content):
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

# Apply, preprocess to the 'content' column


news_dataset['content'] = news_dataset['content'].apply(stemmming)

# To obtain the preprocessed 'content' and label (Y) after preprocessing
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Creating a TF-IDF vectorize to convert text into numerical features
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

# Converting the preprocessed text into TF-IDC features
X = vectorizer.transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=1
)


# Create a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)

# Train the Random Forest model
rf_model = rf.fit(X_train, Y_train)

# Save the trained model and vectorizer
joblib.dump(rf_model, 'trained_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')


# Make predictions on the test set
y_prediction = rf_model.predict(X_test)

# Calculating precision, recall, F1-Score and Accuracy
print("accuracy: ", accuracy_score(Y_test, y_prediction) * 100)
# Making the Confusion Matrix

cm = confusion_matrix(Y_test, y_prediction)
class_label = [0, 1]
df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)

# Plotting the confusion matrix
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
