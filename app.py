from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    
    prediction = model.predict([[age]])
    
    return render_template('result.html', prediction_text=f"Charges = {prediction[0]}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)