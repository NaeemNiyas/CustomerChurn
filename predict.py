import numpy as np
import pickle

# Load the trained XGBoost model
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))

def predict_churn(input_data):
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make predictions
    prediction = xgb_model.predict(input_data_reshaped)

    # Return the prediction (1 for churn, 0 for not churn)
    return prediction[0]

# Example usage:
input_data = [29, 0, 2, 5, 85.47, 460]  # Replace with your own input data
prediction = predict_churn(input_data)

if prediction == 1:
    print("Churn: Yes")
else:
    print("Churn: No")
