import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model_path = 'model/rental_prediction_model.pkl'

    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found. Please train and save the model first.'}), 500

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Read JSON input from request
    user_input = request.json
    Room = user_input.get('Room')
    sq_ft = user_input.get('sq_ft')

    if Room is None or sq_ft is None:
        return jsonify({'error': 'Missing input parameters: Room and sq_ft are required.'}), 400

    try:
        user_input_array = np.array([[float(Room), float(sq_ft)]])
        predicted_rental_price = model.predict(user_input_array)
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

    output = {'Rental Price Prediction Using Model': float(predicted_rental_price[0])}

    # Save output to a JSON file
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/outputs.json', 'w') as f:
        json.dump(output, f)

    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
