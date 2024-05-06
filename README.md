# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Load the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Hema dharshini N
RegisterNumber:  212223220034
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 /(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred 

y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```

## Output:
Dataset

![image](https://github.com/sasirath13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568449/ce3017de-a368-4a72-816e-aca2f9913aa0)

Dataset.dtypes


![image](https://github.com/sasirath13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568449/d809bfe6-b395-4621-a106-598964c6c35a)

Dataset


![image](https://github.com/sasirath13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568449/62d66f05-ec34-4984-9d2e-ea1838edc187)


Y

![image](https://github.com/sasirath13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568449/38f27ffa-8512-46ff-8a4c-30e3176d7c1b)


Accuracy

![image](https://github.com/sasirath13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568449/5350e3ea-08e0-49bb-9cb6-59afa74f0ddc)

Y_pred

![image](https://github.com/sasirath13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568449/26a0182a-3b89-4472-a225-30b8255d046f)

Y

![image](https://github.com/sasirath13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568449/15c5a60d-cb38-42ea-95a9-0f5fe0fcd369)


Y_prednew

![image](https://github.com/sasirath13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568449/70938780-7606-43af-9b93-c959c8a29edf)

Y_prednew

![image](https://github.com/sasirath13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568449/81e8c44b-2fac-4c02-8ba8-d70c1c02c5b7)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

