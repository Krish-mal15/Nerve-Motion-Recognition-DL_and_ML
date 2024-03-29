import pandas as pd

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = pd.read_csv('EMG-data.csv')
data = data.drop(['label', 'time'], axis=1)

X_train = data.drop('class', axis=1).values
y_train = data['class'].values

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# print(data.head())

model = RandomForestClassifier()  # Default parameters

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = (accuracy_score(y_test, predictions)) * 100
print("Accuracy:", accuracy)

joblib.dump(model, 'RFC_ML_Model.joblib')

# 98.27% ACCURACY!!!
