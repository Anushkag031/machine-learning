import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#data collection and analysis
#loading the datsets

diabetes = pd.read_csv('/Users/anushkag031/Documents/machinelearning/diab-pred/diabetes.csv')

#printing the rows
print(diabetes.head())
#entries in the model
print(diabetes.shape)

print("stats/measures : ", diabetes.describe())

# 0 - non-diabetic, 1 - diabetic
print(diabetes['Outcome'].value_counts()) 

#gives mean values for both these cases
print(diabetes.groupby('Outcome').mean())

#separating data and lables
X=diabetes.drop(columns='Outcome',axis=1)
Y=diabetes['Outcome']

print("dropping",X)
print("Y",Y)

#data standardization
scaler=StandardScaler()

#fitting the model
scaler.fit(X)

standardized_values=scaler.transform(X) #to get data in same range

print(standardized_values)

X=standardized_values #saving the standardized values
Y=diabetes['Outcome']

print("X :",X)
print("Y :",Y)


#train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2) #stratify is used to maintain the ratio of 0 and 1 in the test and train data
#random_state is used to get the same data everytime we run the code

print(X.shape, X_train.shape, X_test.shape) #output : (768, 8) (614, 8) (154, 8)

#training the model
classifier=svm.SVC(kernel='linear') #linear kernel

#training the support vector machine
classifier.fit(X_train,Y_train)

#evaluation of the model
#1. accuracy score

#accuracy score on training data
X_train_accuracy=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_accuracy,Y_train) # training data accuracy, using trained ml model

print("Training accuracy score : ",training_data_accuracy)

#accuracy score on test data (imp)
X_test_accuracy=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_accuracy,Y_test) # test data accuracy, using trained ml model

print("Test accuracy score : ",test_data_accuracy)

#making a predictive system
input_data=(5,166,72,19,175,25.8,0.587,51)

#changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1) # -1 is used to get the data in the same format

#standardize the input data
std_data=scaler.transform(input_data_reshaped)
print("std data : ",std_data)

prediction=classifier.predict(std_data)
print("prediction : ",prediction)

if (prediction[0]==0): # prediction[0] is used to get the value from the array (prediction[] is an array)
    print("The person is not diabetic")
else:
    print("The person is diabetic")

#output : The person is diabetic




