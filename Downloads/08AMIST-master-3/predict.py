import pandas as pd
import numpy as np
import json
import pickle
from sklearn.linear_model import LinearRegression
from flask import Flask

app = Flask(__name__)

@app.route('/train', methods=['GET'])
def predict():
    
    # This part is commented out until real training data is defined
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Save the model to disk (if already trained)
    filename = 'model/rental_prediction_model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # Load the model from disk
    filename = 'model/rental_prediction_model.pkl'
    model = pickle.load(open(filename, 'rb'))

    # Read inputs from inputs/inputs.json
    with open('inputs/inputs.json', 'r') as f:
        user_input = json.load(f)

    Room = user_input['Room']
    sq_ft = user_input['sq_ft']

    user_input_prediction = np.array([[Room, sq_ft]])
    predicted_rental_price = model.predict(user_input_prediction)

    # Predict the rental price
    output = {'Rental Price Prediction Using Model': float(predicted_rental_price[0])}

    # Write outputs to outputs/outputs.json
    with open('outputs/outputs.json', 'w') as f:
        json.dump(output, f)

    return output

if __name__ == '__main__':
    app.run(debug=True)
