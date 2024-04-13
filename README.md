# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Read the given dataset.
2.Fitting the dataset into the training set and test set.
3.Applying the feature scaling method.
4.Fitting the logistic regression into the training set.
5.Prediction of the test and result
6.Making the confusion matrix
7.Visualizing the training set results.
```
## Program:
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data = np.loadtxt("ex2data1.txt",delimiter=',')
X = data[:,[0,1]]
y = data[:,2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()
def costFunction(theta,X,Y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T,h-y) / X.shape[0]
  return J,grad
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
J, grad  = costFunction(theta,X_train,y)
print(J)
print(grad)
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
J, grad  = costFunction(theta,X_train,y)
print(J)
print(grad)
def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
  x_min ,x_max = X[:,0].min()-1,X[:,0].max()+1
  y_min ,y_max = X[:,1].min()-1,X[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                      np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
plotDecisionBoundary(res.x,X,y)
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)

```

## Output:
![s1](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/a1aa857b-667d-47d0-a203-3af2f90c23a3)

![s2](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/ef3be58b-041d-440a-acbb-92a1f1d11fc8)

![s3](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/607a1bf4-7ae5-4c83-ae99-6609d2bbaf57)

![s4](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/a2c30a12-c451-4543-8d08-f00429724e71)

![s5](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/1698f8e1-0eeb-4af0-8130-e5f4dcfadcac)

![s6](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/b4fa7365-79bc-4dfb-bca7-a63e3fa99803)

![s7](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/c1c0ea8a-e6eb-43d0-a5ef-7e691185af93)

![s8](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/7ddd432c-7bca-4f11-becc-3a5a4270c9f3)

![s9](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/42e4e669-4c4f-4fcd-8e9b-2a3e1efaed32)

![s10](https://github.com/swetha23013979/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/153823422/844e0cb2-bf68-46de-92a1-d33c1e22538a)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

