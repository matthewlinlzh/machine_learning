from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix

# loading data for train data and test data
train_dataSet = pd.read_csv('Titanic_survivor/train.csv', )
test_dataSet = pd.read_csv('Titanic_survivor/test.csv')

# remove the non-use column and perpare for the data
train_dataSet.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_dataSet.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# fill the missing value with an initial value
train_dataSet = train_dataSet.dropna()
test_dataSet = test_dataSet.dropna()

# preparing the training label and the training data for modeling
features = train_dataSet.columns[1:]
train_dataSet['Sex'] = pd.factorize(train_dataSet['Sex'])[0]
train_dataSet['Embarked'] = pd.factorize(train_dataSet['Embarked'])[0]
labels = train_dataSet['Survived']

# getting the test table clean up for prediction
test_dataSet['Sex'] = pd.factorize(test_dataSet['Sex'])[0]
test_dataSet['Embarked'] = pd.factorize(test_dataSet['Embarked'])[0]
y_true = test_dataSet['Survived']

# using the RandomForestClassifier from ensemble package to predict the model
model = RandomForestClassifier(n_jobs=5, n_estimators=100, max_depth=7, random_state=0)
model.fit(train_dataSet[features], labels)
y_pred = model.predict(test_dataSet[features])

test_dataSet.drop(['Survived'], axis=1, inplace=True)

# make a prediction probability for the prediction
prediction = model.predict_proba(test_dataSet[features])


# create the confusion matrix to see the error rate
confMat = confusion_matrix(y_true, y_pred)
print(confMat)