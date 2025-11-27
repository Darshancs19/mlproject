from flask import Flask, jsonify, request, render_template 
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
# app = application

# import pickle
ridge_model = pickle.load(open('models/model.pkl', 'rb'))
standard_scaler = pickle.load(open('models/preprocessor.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html'), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0")