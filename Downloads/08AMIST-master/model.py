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
def train():
    #Create Pandas DataFrame from CSV file
    df = pd.read_csv('rent2.csv')

    #Features and Labels
    x = df[['Room', 'sq_ft']].values
    y = df['price'].values

    #Split the dataset into training and testing sets
    x_test,x_train,y_test,y_train =train_test_split(x,y,test_size=0.2,random_state=42)


    #Algorithm Selection - Linear Regression
    model= LinearRegression()
    model.fit(x_train,y_train)

    #save the model to disk
    filename = 'model/rental_prediction_model.pkl'
    pickle.dump(model,open(filename, 'wb'))

    # Evaluate the model
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_pred =  model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('mean square error:', mse)

    return('model trained successfully')

@app.route('/predict', methods=['GET'])   
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
 
print(output) 
if __name__ =='__main__':
    app.run(port=5000, debug=True)