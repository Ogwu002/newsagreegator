
import numpy as np
import json
import pickle

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


# write output to outputs/outputs.json
with open('outputs/outputs.json', 'w') as f:
    json.dump(output, f)
