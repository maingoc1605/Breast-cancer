import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data = pd.read_csv("data.csv")
#print(data.info())
Y = data['diagnosis']
def output(x):
    if (x == "M"):
        return 1
    else:
        return 0
Y = data['diagnosis'].apply(output).values
#Y=Y.reshape(-1,1)
print(Y.shape[0])
X=data.drop(['Unnamed: 32', 'id','diagnosis'], axis = 1)
X=X.values
#X=X.T
#print(X.shape[0])
print(X.shape)
m, n = X.shape
X_train = X[:int(0.8 * m), :]
print(X_train.shape)
X_validation = X[int(0.8* m):int(0.9* m), :]
X_test = X[int(0.9 * m):, :]

y_train = Y[:int(0.8 * m)]
print(y_train.shape)
y_validation = Y[int(0.8 * m):int(0.9 * m)]
y_test = Y[int(0.9 * m):]
np.random.seed(1)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def costfunc(theta,X,Y,lambda_=0.1):
    m=Y.size
    h=sigmoid(np.dot(X,theta.T))
    J=(-1/m)*(np.dot(Y,np.log(h))+ np.dot((1-Y),np.log(1-h)))
    reg= (lambda_/2*m)*(np.sum(np.square(theta[1:])))
    J=J+reg
    return J
def Gradient(theta,X,y,lambda_=0.1):
    m=y.size
    grad=(1/m) *np.dot(sigmoid(np.dot(X,theta.T))-y,X)
    grad[1:] = grad[1:] + (lambda_ / m )* theta[1:]
    return grad

def calculateTheta(X, y, lambda_):
    if X.ndim == 1:
        X = X[None]
    options = {'maxiter': 50}
    thetas = np.zeros((X.shape[1],1))
    res = optimize.minimize(costfunc,
                            thetas,
                            (X, y, lambda_),
                            jac=Gradient,
                            method='TNC',
                            options=options)
    theta = res.x
    return theta

def predict(theta, X):
    result = sigmoid(np.dot(theta, X.T))
    if result >= 0.5:
        return 1, result
    else:
        return 0, result
lambda_ = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 50]
#tim theta tot nhat vi cost nho nhat
all_theta = np.zeros((len(lambda_), X.shape[1]))
print(all_theta.shape)
for i in range(len(lambda_)):
    theta = calculateTheta(X_train, y_train, lambda_[i])
    print("lambda uses:" + str(lambda_[i]))
    print("Theta found: " + str(theta))
    J_test = costfunc(theta, X_test, y_test, lambda_=0)
    print("Cost: " + str(J_test))
    all_theta[i] = theta
best_J_cv = costfunc(all_theta[0], X_validation, y_validation, lambda_=0)
chosen_theta = np.zeros((X.shape[1], 1))
print("\n \n")
for i in range(all_theta.shape[0]):
    temp = costfunc(all_theta[i], X_validation, y_validation, lambda_=0)
    print("Theta use: " + str(all_theta[i]))
    print("Cost calculated: " + str(temp))
    if temp <= best_J_cv:
        best_J_cv = temp
        chosen_theta = all_theta[i]
print("Best validation cost: " + str(best_J_cv))
print("Best theta: " + str(chosen_theta))
print("\n \n")
#activation function: sigmoid 
# evaluate model
right = 0
wrong = 0
test_cost = costfunc(chosen_theta, X_test, y_test, lambda_=0)
print("Test cost: " + str(test_cost))
for i in range(y_test.size):
    result = predict(chosen_theta, X_test[i])
    print("Result by model: " + str(result) + "\n" + "Actual result: " + str(y_test[i]))
    if result[0] == y_test[i]:
        right += 1
    else:
        wrong += 1
print("\n \n")
print("right: " + str(right))
print("Total: "+ str(y_test.size))
print("accuracy: " + str(right/y_test.size))


