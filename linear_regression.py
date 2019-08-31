import numpy
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("data.csv")
X = dataset['sqft_living']
Y = dataset['price']
X = [X]
Y = [Y]

regressor = LinearRegression()

regressor.fit(X,Y)
plt.scatter(X,Y,color="red")
plt.show()
