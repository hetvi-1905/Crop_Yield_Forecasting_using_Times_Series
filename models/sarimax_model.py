from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def predict_sarimax(data):
    # Ensure data is numeric and drop NaNs
    data['Yield'] = pd.to_numeric(data['Yield'], errors='coerce')
    data = data.dropna()

    # Fit the ARIMAX model
    model = SARIMAX(data['Yield'], order=(5, 1, 0))  # Adjust the order as necessary
    model_fit = model.fit()

    # Forecast the next 5 years
    forecast = model_fit.forecast(steps=5)  # Provide last available exogenous values
    
    # Create a Series for the predictions with the correct index
    forecast_years = range(data.index[-1] + 1, data.index[-1] + 6)
    predictions = pd.Series(forecast, index=forecast_years)

    return predictions

# Load your dataset and exogenous variables
df = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\Crop Yield Forecasting\data\maize_yield_data.csv')  # Update with your actual CSV file path
# exogenous = pd.read_csv('path_to_exogenous_variables.csv')  # Update with actual exogenous data

df.set_index('Year', inplace=True)

# Call the prediction function
predictions = predict_sarimax(df)
print("SARIMAX Predicted yields for the next 5 years:")
print(predictions)
