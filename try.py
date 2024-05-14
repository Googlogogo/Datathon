import pandas as pd
import numpy as np
import joblib
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import *
from sklearn import metrics
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('heart_train.csv')
df['Sex'].replace(['M', 'F'], [1, 0], inplace=True)
df['ChestPainType'].replace(['ATA', 'NAP', 'ASY', 'TA'], [0, 1, 2, 3], inplace=True)
df['RestingECG'].replace(['Normal', 'ST', 'LVH'], [0, 1, 2], inplace=True)
df['ExerciseAngina'].replace(['N', 'Y'], [0, 1], inplace=True)
df['ST_Slope'].replace(['Flat', 'Up', 'Down'], [0, 1, 2], inplace=True)
X = df.drop(columns='HeartDisease')
y = df['HeartDisease']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions))
print('Recall: ', recall_score(y_test, predictions, zero_division=1))
print('Precision: ', precision_score(y_test, predictions, zero_division=1))
print('CL Report:\n', classification_report(y_test, predictions, zero_division=1))
joblib.dump(model, "heart_disease_model.joblib")
test_model = joblib.load("heart_disease_model.joblib")
test_df = pd.read_csv('heart_test.csv')
test_df['Sex'].replace(['M', 'F'], [1, 0], inplace=True)
test_df['ChestPainType'].replace(['ATA', 'NAP', 'ASY', 'TA'], [0, 1, 2, 3], inplace=True)
test_df['RestingECG'].replace(['Normal', 'ST', 'LVH'], [0, 1, 2], inplace=True)
test_df['ExerciseAngina'].replace(['N', 'Y'], [0, 1], inplace=True)
test_df['ST_Slope'].replace(['Flat', 'Up', 'Down'], [0, 1, 2], inplace=True)
test_predictions = test_model.predict(test_df)
output = pd.DataFrame({"id": test_df["id"], "output": test_predictions}).to_csv('heart_disease_submission.csv')