import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv("taxi.csv")
#print(data.head())
x=data.iloc[:,0:-1].values
y=data.iloc[:,-1].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
reg = LinearRegression()
reg.fit(x_train,y_train)
pickle.dump(reg,open("taxi.pkl","wb"))
model=pickle.load(open("taxi.pkl","rb"))
print(model.predict([[80,1770000,6000,85]]))
