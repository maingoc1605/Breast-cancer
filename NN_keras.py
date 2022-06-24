import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data = pd.read_csv("data.csv")
#rint(data.head())
def output(x):
    if x=="M":
        return 1
    else:
        return 0
y= data['diagnosis'].apply(output)
#print(y.shape)
X=data.drop(['Unnamed: 32', 'id','diagnosis'], axis = 1)#569 sample, 30 feature\
m,n=X.shape
#print(X.shape)
#split data: 80% for train, 10% for validation, 10% for test
X_train=X.iloc[:int(0.8*m),:]
#print(X_train.shape)
y_train=y[:int(0.8*m)]

X_val = X.iloc[int(0.8* m):int(0.9* m), :]
X_test = X.iloc[int(0.9 * m):, :]
y_val = y[int(0.8 * m):int(0.9 * m)]
y_test = y[int(0.9 * m):]

tf.random.set_seed(42)
model= tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
    #tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(lr=0.01),
            metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=23,validation_data=(X_val,y_val))
def plot_learing_curve(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(loss))
    # plot loss
    plt.plot(epochs, loss, label='training loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.xlabel('Epochs')
    plt.legend()
    # plt accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training accuracy')
    plt.plot(epochs, val_accuracy, label='val accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

plot_learing_curve(history)
#y_pred=model.predict(X_test)
result=[]
error=0
y_pred = model.predict(X_test)
y_pred=np.round(y_pred)
print(y_pred)
#print(y_test)
y_test=y_test.values
for i in range(len(y_pred)):
    print("Result by NN: " + str(y_pred[i]) + "\n" + "Actual result: " + str(y_test[i]))
    if (y_pred[i] != y_test[i]):
        error= error+1
print (error)



