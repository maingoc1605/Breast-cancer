import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("data.csv")
#print(data.info())
Y = data['diagnosis']
def output(x):
    if (x == "M"):
        return 1
    else:
        return 0
Y = data['diagnosis'].apply(output).values

print(Y.shape)
X=data.drop(['Unnamed: 32', 'id','diagnosis'], axis = 1)
print(X.shape)
np.random.seed(1)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=10000)
model.fit(X_train,Y_train)
y_pred =model.predict(X_test)
print("coef: " + str(model.coef_))
print("intercept: " + str(model.intercept_))
print("Score: " + str(model.score(X_test, Y_test)))

result=[]
result = model.predict(X_test)
for i in range(len(result)):
    print("Result by sklearn: " + str(result[i]) + "\n" + "Actual result: " + str(Y_test[i]))
