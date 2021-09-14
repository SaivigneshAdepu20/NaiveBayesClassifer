# Importing Library
import pandas as pd

# Importing the dataset  
df = pd.read_csv('DataSet.csv')  
print("\nOriginal Dataframe :\n" + str(df))

# Classifier Column
target = df.buys_computer
print("\nClassifier Column :\n" + str(target))

# Required Dataframe
inputs = df.drop(['rid', 'buys_computer'], axis=1)
print("\nRequired Dataframe (Feature Columns) :\n" + str(inputs))

# Converting String Data Type to Interger Data Type
dummies_1 = pd.get_dummies(inputs.age)
print("\nAge Dummy Values :\n" + str(dummies_1))
dummies_2 = pd.get_dummies(inputs.income)
print("\nIncome Dummy Values :\n" + str(dummies_2))
dummies_3 = pd.get_dummies(inputs.student)
print("\nStudent Dummy Values :\n" + str(dummies_3))
dummies_4 = pd.get_dummies(inputs.credit_rating)
print("\nCredit_Rating Dummy Values :\n" + str(dummies_4))

# Getting Dummy valued Columns
inputs = pd.concat([inputs, dummies_1, dummies_2, dummies_3, dummies_4], axis=1)
print("\nConcatinating ALL Columns :\n" + str(inputs))
inputs.drop(['age', 'income', 'student', 'credit_rating'], axis=1, inplace=True)
print("\nRemoving Columns with \"String Data Type\" :\n" + str(inputs))

# Filling Empty Values
print("\nColumn names with \"Empty Values\" :\n" + str(inputs.columns[inputs.isna().any()]))
for i in ['middleage', 'senior', 'youth', 'high', 'low', 'medium', 'no', 'yes', 'excellent', 'fair']:
    inputs[i] = inputs[i].fillna(inputs[i].mean())

# Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.25, random_state = None)  
print("\nX_test data :\n" + str(x_test))
print("\nY_test data :\n" + str(y_test))

# Fitting Naive Bayes to the Training set  
from sklearn.naive_bayes import GaussianNB  
model = GaussianNB()  
model.fit(x_train, y_train)  

# Predicting the Test set results  
y_pred = model.predict(x_test)  
print("\nY_Predictions :\n" + str(y_pred))

# Model Accuracy
model_score = model.score(x_test, y_test)
print("\nModel Accuracy : " + str(model_score * 100) + " %")

# Making the Confusion Matrix  
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)  
print("\nConfusion Matrix :\n" + str(cm))

# Query_1 = ['youth', 'high', 'no', 'fair']
# Query_1 Output = 'no'
y_pred = model.predict([[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]])
print("\nQuery_1 = ['youth', 'high', 'no', 'fair']")
print("Query_1 Actual Output : ['no']")
print("Predicted Output : " + str(y_pred))

# Query_2 = ['youth', 'high', 'no', 'excellent']
# Query_2 Output = 'no'
y_pred = model.predict([[0, 0, 1, 1, 0, 0, 1, 0, 1, 0]])
print("\nQuery_2 = ['youth', 'high', 'no', 'excellent']")
print("Query_2 Actual Output : ['no']")
print("Predicted Output : " + str(y_pred))

# Query_3 = ['middleage', 'high', 'no', 'fair']
# Query_3 Output = 'yes'
y_pred = model.predict([[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]])
print("\nQuery_3 = ['middleage', 'high', 'no', 'fair']")
print("Query_3 Actual Output : ['yes']")
print("Predicted Output : " + str(y_pred) + "\n")
