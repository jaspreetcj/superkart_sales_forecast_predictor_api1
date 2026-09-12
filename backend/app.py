# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_forecast_predictor_api = Flask("Jaspreet's SuperKart sales forecast predictor")

# Load the trained machine learning model
model = joblib.load("best_random_forest_model.joblib")

# Define a route for the home page (GET request)
@superkart_sales_forecast_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Jaspreet's SuperKart sales forecast Prediction API!"

# Define an endpoint for single property prediction (POST request)
@superkart_sales_forecast_predictor_api.post('/v1/salesForecast')
def predict_sales_forecast_price():
    """
    This function handles POST requests to the '/v1/salesForecast' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted rental price as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'product_weight': product_data['Product_Weight'],
        'product_sugar_content': product_data['Product_Sugar_Content'],
        'product_allocated_area': product_data['Product_Allocated_Area'],
        'product_mrp': product_data['Product_MRP'],
        'product_id_char': product_data['Product_Id_char'],
        'product_type_category': product_data['Product_Type_Category'],
        'store_size': product_data['Store_Size'],
        'store_location_city_type': product_data['Store_Location_City_Type'],
        'store_type': product_data['Store_Type'],
        'store_age_years': product_data['Store_Age_Years']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction 
    predicted_sales_target = model.predict(input_data)[0]

    # Return the actual sales target
    return jsonify({'Predicted sales target ': predicted_sales_target})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_forecast_predictor_api.post('/v1/salesTargetBatch')
def predict_rental_price_batch():
    """
    This function handles POST requests to the '/v1/salesTargetBatch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame
    predicted_saleT= model.predict(input_data).tolist()

    # Create a dictionary of predictions with  IDs as keys
    ids = input_data['id'].tolist()  # Assuming 'id' is the property ID column
    output_dict = dict(zip(ids, predicted_saleT))  # Use actual prices

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_forecast_predictor_api.run(debug=True)
