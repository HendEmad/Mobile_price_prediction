from flask import Flask, request
from joblib import load

app = Flask(__name__)

# Load the saved model when the application starts
loaded_model = load('models/model.pkl')


# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json['data']

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)

    # Return the predictions as a JSON response
    return {'predictions': predictions.tolist()}


if __name__ == '__main__':
    app.run()