import pandas as pd
import numpy as np
import json
import pickle
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from flask import Flask


app = Flask(__name__)

@app.route('/train', methods=['GET'])
def predict():

#load the model from disk
filename = 'model/rental_prediction_model.pkl'
model = pickle.load(open(filename, 'rb'))

#Read Inputs from inputs/inputs.json
with open('inputs/inputs.json', 'r') as f:
 user_input = json.load(f)

Room =  user_input['Room']
sq_ft =  user_input['sq_ft']

user_input_prediction = np.array([[Room,sq_ft]])
predicted_rental_price = model.predict(user_input_prediction )
#predict the rental price
output = {'Rental Price Prediction Using Model':float(predicted_rental_price[0])}


# write outputs to outputs/outputs.json
with open('outputs/outputs.json', 'w') as f:
    json.dump(output, f)
