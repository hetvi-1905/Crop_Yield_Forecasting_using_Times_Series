# import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
# import numpy as np

# def predict_arima(data):
#     # Ensure data is in numeric format and drop NaNs
#     data['Yield'] = pd.to_numeric(data['Yield'], errors='coerce')
#     data = data.dropna()

#     # Fit the ARIMA model
#     model = ARIMA(data['Yield'], order=(5, 1, 0))  # Adjust the order as necessary
#     model_fit = model.fit()

#     # Forecast the next 5 years
#     forecast = model_fit.forecast(steps=5)
    
#     # Create a Series for the predictions with the correct index
#     forecast_years = range(data.index[-1] + 1, data.index[-1] + 6)
#     predictions = pd.Series(forecast, index=forecast_years)

#     return predictions

# # Load your dataset
# df = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\Crop Yield Forecasting\data\maize_yield_data.csv')  # Update with your actual CSV file path
# df.set_index('Year', inplace=True)

# # Call the prediction function
# predictions = predict_arima(df)
# print("ARIMA Predicted yields for the next 5 years:")
# print(predictions)


import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def predict_arima(data):
    """Predicts crop yield using an ARIMA model."""
    # Ensure the 'Yield' column is numeric
    data['Yield'] = pd.to_numeric(data['Yield'], errors='coerce')
    data = data.dropna()

    # Fit the ARIMA model (you may need to adjust the order based on your dataset)
    model = ARIMA(data['Yield'], order=(5, 1, 0))  # Example order; adjust as needed
    model_fit = model.fit()

    # Forecast the next 5 years
    forecast = model_fit.forecast(steps=5)

    # Get the last year from the input data
    last_year = data.index[-1]
    forecast_years = range(last_year + 1, last_year + 6)  # Next 5 years

    # Create a Series for the predictions with the correct index
    predictions_series = pd.Series(forecast, index=forecast_years)

    return predictions_series

# Load data from a CSV file
data_file_path = r'C:\Users\Admin\OneDrive\Desktop\Crop Yield Forecasting\data\maize_yield_data.csv'  # Update with your actual CSV file path
df = pd.read_csv(data_file_path)  # Make sure the CSV contains 'Year' and 'Yield' columns
df.set_index('Year', inplace=True)

# Call the prediction function
try:
    predictions = predict_arima(df)
    print("ARIMA Predicted yields for the next 5 years:")
    print(predictions)
except Exception as e:
    print("Error during ARIMA prediction:", e)
