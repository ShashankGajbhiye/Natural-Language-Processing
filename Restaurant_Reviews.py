
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the Texts in the Dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the Dataset in Training Set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Creating Function to check which model has higher Accuracy
def models(x_train, y_train):
    
    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
    forest.fit(x_train, y_train)

    # Navie Bayes Model
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    
    # Printing the Accuracy of the Models
    print('[0] Random Forest Classifier Accuracy:', forest.score(x_train, y_train))
    print('[0] Naive Bayes Classifier Accuracy:', nb.score(x_train, y_train))
    
    return forest, nb

# Getting All Models
model = models(x_train, y_train)

"""
# Predicting the Test Set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm
acc = accuracy_score(y_test, y_pred) * 100
acc
"""
# Creating the Classification Report
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
for i in range(len(model)):
    print('Model ', i)
    print(classification_report(y_test, model[i].predict(x_test)))
    print('Accuracy:', accuracy_score(y_test, model[i].predict(x_test)))
    print(confusion_matrix(y_test, model[i].predict(x_test)))
    print()

# Predictions of the Naive Bayes Classifier
y_pred = model[1].predict(x_test)
y_pred