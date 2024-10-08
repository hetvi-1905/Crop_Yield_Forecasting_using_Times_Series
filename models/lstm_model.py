# # # import numpy as np
# # # import pandas as pd
# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import LSTM, Dense, Dropout
# # # from sklearn.preprocessing import MinMaxScaler

# # # def create_lstm_model(input_shape):
# # #     model = Sequential()
# # #     model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
# # #     model.add(Dropout(0.2))
# # #     model.add(LSTM(units=50, return_sequences=False))
# # #     model.add(Dropout(0.2))
# # #     model.add(Dense(units=1))
# # #     model.compile(optimizer='adam', loss='mean_squared_error')
# # #     return model

# # # def predict_lstm(data):
# # #     # Prepare data
# # #     data = data.dropna()
# # #     data_scaled = MinMaxScaler().fit_transform(data['Yield'].values.reshape(-1, 1))
    
# # #     # Reshape for LSTM input
# # #     X_train = np.array([data_scaled[i-10:i] for i in range(10, len(data_scaled))])
# # #     y_train = data_scaled[10:]

# # #     # Build the LSTM model
# # #     model = create_lstm_model((X_train.shape[1], 1))
# # #     model.fit(X_train, y_train, epochs=10, batch_size=32)

# # #     # Predict the next 5 years
# # #     last_10_days = data_scaled[-10:].reshape(1, 10, 1)
# # #     forecast = model.predict(last_10_days)

# # #     return pd.Series(forecast.flatten(), index=range(data.index[-1] + 1, data.index[-1] + 6))
# # import numpy as np
# # import pandas as pd
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import LSTM, Dense, Dropout
# # from sklearn.preprocessing import MinMaxScaler

# # def create_lstm_model(input_shape):
# #     """Creates and compiles the LSTM model."""
# #     model = Sequential()
# #     model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
# #     model.add(Dropout(0.2))
# #     model.add(LSTM(units=50, return_sequences=False))
# #     model.add(Dropout(0.2))
# #     model.add(Dense(units=1))
# #     model.compile(optimizer='adam', loss='mean_squared_error')
# #     return model

# # def predict_lstm(data):
# #     """Predicts crop yield using an LSTM model."""
# #     # Ensure that 'Yield' column is numeric
# #     data['Yield'] = pd.to_numeric(data['Yield'], errors='coerce')

# #     # Drop any rows with NaN values
# #     data = data.dropna()

# #     # Prepare data for LSTM
# #     scaler = MinMaxScaler()
# #     data_scaled = scaler.fit_transform(data['Yield'].values.reshape(-1, 1))

# #     # Ensure data length is sufficient for LSTM
# #     if len(data_scaled) < 10:
# #         raise ValueError("Insufficient data: At least 10 data points are required for LSTM.")

# #     # Reshape data for LSTM input
# #     X_train = np.array([data_scaled[i-10:i] for i in range(10, len(data_scaled))])
# #     y_train = data_scaled[10:]

# #     # Build and train the LSTM model
# #     model = create_lstm_model((X_train.shape[1], 1))
# #     model.fit(X_train, y_train, epochs=10, batch_size=1)  # Use batch size of 1 for smaller datasets

# #     # Predict the next 5 years
# #     last_10_days = data_scaled[-10:].reshape(1, 10, 1)
# #     forecast = model.predict(last_10_days)

# #     # Inverse transform the forecast to get back to original scale
# #     forecast_inverse = scaler.inverse_transform(forecast)

# #     # Create a Series for the predictions with the correct index
# #     forecast_years = range(data.index[-1] + 1, data.index[-1] + 6)  # Predict for the next 5 years
# #     predictions = pd.Series(forecast_inverse.flatten(), index=forecast_years)

# #     return predictions

# # # Load data from a CSV file
# # df = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\Crop Yield Forecasting\data\maize_yield_data.csv')  # Update with your actual CSV file path
# # df.set_index('Year', inplace=True)  # Assuming 'Year' is the first column

# # # Ensure 'Yield' is in numeric format and handle errors
# # df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')

# # # Drop any rows with NaN values
# # df = df.dropna()

# # # Call the prediction function
# # try:
# #     predictions = predict_lstm(df)
# #     print("Predicted yields for the next 5 years:")
# #     print(predictions)
# # except ValueError as e:
# #     print("Error during prediction:", e)
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler

# def create_lstm_model(input_shape):
#     """Creates and compiles the LSTM model."""
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# def predict_lstm(data):
#     """Predicts crop yield using an LSTM model."""
#     # Ensure that 'Yield' column is numeric
#     data['Yield'] = pd.to_numeric(data['Yield'], errors='coerce')
#     data = data.dropna()

#     # Prepare data for LSTM
#     scaler = MinMaxScaler()
#     data_scaled = scaler.fit_transform(data['Yield'].values.reshape(-1, 1))

#     # Ensure data length is sufficient for LSTM
#     if len(data_scaled) < 10:
#         raise ValueError("Insufficient data: At least 10 data points are required for LSTM.")

#     # Reshape data for LSTM input
#     X_train = np.array([data_scaled[i-10:i] for i in range(10, len(data_scaled))])
#     y_train = data_scaled[10:]

#     # Build and train the LSTM model
#     model = create_lstm_model((X_train.shape[1], 1))
#     model.fit(X_train, y_train, epochs=50, batch_size=1)  # Use more epochs for better training

#     # Predict the next 5 years
#     last_10_days = data_scaled[-10:].reshape(1, 10, 1)
#     forecast = model.predict(last_10_days)

#     # Inverse transform the forecast to get back to original scale
#     forecast_inverse = scaler.inverse_transform(forecast)

#     # Create a Series for the predictions with the correct index
#     forecast_years = range(data.index[-1] + 1, data.index[-1] + 6)  # Predict for the next 5 years
#     predictions = pd.Series(forecast_inverse.flatten(), index=forecast_years)

#     return predictions

# # Load data from a CSV file
# data_file_path = r'C:\Users\Admin\OneDrive\Desktop\Crop Yield Forecasting\data\maize_yield_data.csv'  # Update with your actual CSV file path
# df = pd.read_csv(data_file_path)  # Make sure the CSV contains 'Year' and 'Yield' columns
# df.set_index('Year', inplace=True)

# # Call the prediction function
# try:
#     predictions = predict_lstm(df)
#     print("LSTM Predicted yields for the next 5 years:")
#     print(predictions)
# except ValueError as e:
#     print("Error during LSTM prediction:", e)
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def create_lstm_model(input_shape):
    """Creates and compiles the LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_lstm(data):
    """Predicts crop yield using an LSTM model."""
    # Ensure that 'Yield' column is numeric
    data['Yield'] = pd.to_numeric(data['Yield'], errors='coerce')
    data = data.dropna()

    # Prepare data for LSTM
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data['Yield'].values.reshape(-1, 1))

    # Ensure data length is sufficient for LSTM
    if len(data_scaled) < 10:
        raise ValueError("Insufficient data: At least 10 data points are required for LSTM.")

    # Reshape data for LSTM input
    X_train = np.array([data_scaled[i-10:i] for i in range(10, len(data_scaled))])
    y_train = data_scaled[10:]

    # Build and train the LSTM model
    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=1)  # Use more epochs for better training

    # Predict the next 5 years
    # Start with the last 10 data points for the prediction
    predictions = []
    current_input = data_scaled[-10:].reshape(1, 10, 1)  # Reshape for LSTM input

    for _ in range(5):
        forecast = model.predict(current_input)  # Make the prediction
        predictions.append(forecast[0, 0])  # Store the forecast
        current_input = np.append(current_input[:, 1:, :], forecast.reshape(1, 1, 1), axis=1)  # Update input

    # Inverse transform the forecast to get back to original scale
    predictions_inverse = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Create a Series for the predictions with the correct index
    forecast_years = range(data.index[-1] + 1, data.index[-1] + 6)  # Predict for the next 5 years
    predictions_series = pd.Series(predictions_inverse.flatten(), index=forecast_years)

    return predictions_series

# Load data from a CSV file
data_file_path = r'C:\Users\Admin\OneDrive\Desktop\Crop Yield Forecasting\data\maize_yield_data.csv'  # Update with your actual CSV file path
df = pd.read_csv(data_file_path)  # Make sure the CSV contains 'Year' and 'Yield' columns
df.set_index('Year', inplace=True)

# Call the prediction function
try:
    predictions = predict_lstm(df)
    print("LSTM Predicted yields for the next 5 years:")
    print(predictions)
except ValueError as e:
    print("Error during LSTM prediction:", e)
