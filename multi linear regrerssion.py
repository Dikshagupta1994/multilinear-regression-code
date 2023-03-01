from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset=pd.read_csv('C:/Users\hp\Downloads\startups.csv')
dataset.head()

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],
                     remainder='passthrough')

x=np.array(ct.fit_transform(x))
print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,
                                               random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)

y_pred=regressor.predict(x_test)
regressor.score(x_train,y_train)
regressor.score(x_test,y_test)

plt.figure(dpi=300)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train))
plt.title('spend money', color='blue')
plt.xlabel('spend money')










