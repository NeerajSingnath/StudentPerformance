from flask import render_template
import joblib
from flask import request
from flask import Flask
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        return "Prediction coming soon"

if __name__ == '__main__':
    app.run(debug=True)
