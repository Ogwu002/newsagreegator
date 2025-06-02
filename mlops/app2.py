import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
from flask import Flask
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/train', methods=['GET'])

def train():

    df = pd.read_csv('rent2.csv')

    x = df[['Room','sq_ft']]
    y = df['price']

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    model = LinearRegression()
    model = model.fit(x_train,y_train)

    #save the model to disk
    filename = 'model/rental_prediction_model.pkl'
    pickle.dump(model,open(filename, 'wb'))

    #Evaluate the model
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return ('model trained successfully')

@app.route('/predict', methods=['GET'])
def predict():
    #load the model from disk
    filename = 'model/rental_prediction_model.pkl'
    model = pickle.load(open(filename, 'rb')) 

    #Read inputs from inputs/inputs.json
    with open('inputs.json', 'r') as f:
        user_input = json.load(f)

    Room = user_input['Room']
    sq_ft = user_input['sq_ft']

    user_input_prediction = np.array([[Room,sq_ft]])
    predicted_rental_price = model.predict(user_input_prediction)

    #predict the rental price
    output = {'Rental Price prediction using model':float(predicted_rental_price[0])}

    #write outputs to outputs.json
    with open('outputs.json', 'w') as f:
        json.dump(output,f)

    print(output)
    return output
if __name__ == '__main__':
    app.run(port=5000, debug=True)

train()
predict()
