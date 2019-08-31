import numpy
import matplotlib.pyplot as plt 
import pandas as pd
#from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("data.csv")
X = dataset['sqft_living']
Y = dataset['price']

#print(X)
#print(Y)

len_x = len(X)
len_y = len(Y)
x_sum = 0 
y_sum = 0
for i in range(0,len_x):
    x_sum = x_sum + X[i]
    y_sum = y_sum + Y[i]

X_mean = x_sum/len_x
Y_mean = y_sum/len_y

#print(Y_mean, X_mean)

slope_num = 0
slope_den = 0

for i in range(0,len_x):
    slope_num = slope_num + (X[i]-X_mean)*(Y[i]-Y_mean)
    slope_den = slope_den + (X[i]-X_mean)**2

slope=slope_num/slope_den

C_term = Y_mean - X_mean*slope

#print(slope, C_term)

Y_new = list()
for i in range(0,len_x):
    Y_new.append(X[i]*slope + C_term)

plt.scatter(X,Y,color="red")
plt.plot(X,Y_new,color="black")
plt.show()
