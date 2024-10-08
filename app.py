from flask import Flask, render_template, request, jsonify
import pandas as pd
from models.arima_model import predict_arima
from models.sarimax_model import predict_sarimax
from models.lstm_model import predict_lstm

app = Flask(__name__)

# Load crop datasets
crop_data = {
    'Wheat': pd.read_csv('data/wheat_yield_data.csv'),
    'Paddy': pd.read_csv('data/paddy_yield_data.csv'),
    'Maize': pd.read_csv('data/maize_yield_data.csv')
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    crop_type = request.form['crop_type']
    model_type = request.form['model_type']
    
    # Fetch the dataset for the selected crop
    crop_data_selected = crop_data[crop_type]

    # Make predictions based on the model type
    if model_type == 'ARIMA':
        result = predict_arima(crop_data_selected)
    elif model_type == 'SARIMAX':
        result = predict_sarimax(crop_data_selected)
    else:
        result = predict_lstm(crop_data_selected)

    # Return results as a JSON response
    return jsonify(result.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
