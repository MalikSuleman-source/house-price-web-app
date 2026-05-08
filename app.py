from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['MedInc']),
        float(request.form['HouseAge']),
        float(request.form['AveRooms']),
        float(request.form['AveBedrms']),
        float(request.form['Population']),
        float(request.form['AveOccup']),
        float(request.form['Latitude']),
        float(request.form['Longitude'])
    ]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    price = round(prediction * 100000, 2)
    return render_template('index.html', 
                         prediction=f"${price:,.0f}")

if __name__ == '__main__':
    app.run(debug=True)