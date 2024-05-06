# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.
2.Load the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value
```
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

![1](https://github.com/hema-dharshini5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147117728/f984ac01-16a6-4e16-a288-73215d99fa8a)


Dataset.dtypes


![2](https://github.com/hema-dharshini5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147117728/7f1ae4c4-7758-445e-beae-4ec8da4e05fa)


Dataset


![3](https://github.com/hema-dharshini5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147117728/8233e8f6-653e-45e5-88f9-38684e127382)



Y

![4](https://github.com/hema-dharshini5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147117728/6e4e721a-00a2-429a-9d3f-b3f9c94be681)


Accuracy

![5](https://github.com/hema-dharshini5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147117728/a6377038-645f-47b6-bf31-3c9e3774c3cb)

Y_pred

![6](https://github.com/hema-dharshini5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147117728/3a586886-1a39-4927-9a07-690bf5b260e7)

Y

![7](https://github.com/hema-dharshini5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147117728/25834f32-967d-434e-8064-b2f51fe183ea)



Y_prednew

![8](https://github.com/hema-dharshini5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147117728/13349b32-60c1-4b53-ab30-ab7382275906)

Y_prednew

![9](https://github.com/hema-dharshini5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147117728/b250e74a-2c29-4709-bf4c-326be5fe860a)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

