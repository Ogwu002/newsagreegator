import pandas as pd
import numpy as np
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
df = pd.read_csv('rent2.csv')
x = df[['Room', 'sq_ft']].values
y = df['price'].values
x_test,x_train,y_test,y_train =train_test_split(x,y,test_size=0.2,random_state=42)
#Algorithm Selection
model= LinearRegression()
model.fit(x_train,y_train)
filename = 'model/rental_prediction_model.pkl'
pickle.dump(model,open(filename, 'wb'))
#Evaluate the model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred =  model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
