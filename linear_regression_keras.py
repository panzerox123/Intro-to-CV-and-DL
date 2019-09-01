import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model,Input
from keras.callbacks import TensorBoard

data = pd.read_csv('data.csv')
x,y = data['sqft_living'],data['price']
plt.scatter(x,y,color="blue")
#plt.show()

model = Sequential()
model.add(Dense(1, input_dim=1, kernel_initializer='normal',activation="linear"))
model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', metrics=['mse'])
model.summary()


tensorboard = TensorBoard(log_dir='./tensor_log')
hist = model.fit(x,y,batch_size=1,epochs=50,shuffle=False,callbacks=[tensorboard])

x_test = np.array(x)
y_test = np.array(y)
score = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
plt.plot(x_test,y_predict)
plt.show()
print("test score: ", score)