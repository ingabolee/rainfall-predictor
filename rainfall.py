import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


#reading the csv file
df = pandas.read_csv("file.csv")


#Defining our data attributes. Independent variables
x = df[['Evaporation', 'Sunshine', 'WindSpeed', 'Humidity', 'Pressure', 'Temperature']]

#Defining our dependent variable
y = df['Rainfall']


#Splitting our data into training and testing sets with an 80/20 percentage
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#initializing our Linear regresion model
regr = linear_model.LinearRegression()


#Trinining the model
regr.fit(x_train, y_train)


#testing the model
y_prediction = regr.predict(x_test)


#printing the model perfomance metrics
mse = mean_squared_error(y_test, y_prediction)
error = sqrt(mse)
print(f"mean squared error is {mse}")
print(f"error value is {error}")

