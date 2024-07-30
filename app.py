from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Define custom classes used in the decision tree model
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val == self.value

    def __repr__(self):
        return f"Is {header[self.column]} == {str(self.value)}?"

class Leaf:
    def __init__(self, data):
        self.predictions = class_counts(data)

class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def class_counts(data):
    counts = {}
    for row in data:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# Load and preprocess data
crop_data = pd.read_csv('Crop_recommendation.csv')
fertilizer_data = pd.read_csv("fertilizer_recommendation.csv")
rainfall_data = pd.read_csv('rainfall_in_india_1901-2015.csv')

# Preprocess fertilizer data
le_soil = LabelEncoder()
le_crop = LabelEncoder()
fertilizer_data['Soil Type'] = le_soil.fit_transform(fertilizer_data['Soil Type'])
fertilizer_data['Crop Type'] = le_crop.fit_transform(fertilizer_data['Crop Type'])

# Train models
def train_crop_model():
    X = crop_data.iloc[:, :-1].values
    y = crop_data.iloc[:, -1].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    return model.fit(X_train, y_train)

def train_fertilizer_model():
    X = fertilizer_data.iloc[:, :8]
    y = fertilizer_data.iloc[:, -1]
    model = DecisionTreeClassifier(random_state=0)
    return model.fit(X, y)

crop_model = train_crop_model()
fertilizer_model = train_fertilizer_model()

# Load decision tree model
header = ['State_Name', 'District_Name', 'Season', 'Crop']
dt_model = joblib.load('filetest2.pkl')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = request.json
    input_data = [data[key] for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    prediction = crop_model.predict([input_data])
    return jsonify({'predicted_crop': prediction[0]})

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    data = request.json
    soil_enc = le_soil.transform([data['soil_type']])[0]
    crop_enc = le_crop.transform([data['crop_type']])[0]
    input_data = [data['temperature'], data['humidity'], data['moisture'], soil_enc, crop_enc, 
                  data['nitrogen'], data['potassium'], data['phosphorous']]
    prediction = fertilizer_model.predict([input_data])
    return jsonify({'predicted_fertilizer': prediction[0]})

@app.route('/predict_rainfall', methods=['POST'])
def predict_rainfall():
    data = request.json
    state_data = rainfall_data[rainfall_data['SUBDIVISION'] == data['state']]
    avg_rainfall = state_data[data['month']].mean()
    return jsonify({'predicted_rainfall': avg_rainfall})

@app.route('/predict_crop_dt', methods=['POST'])
def predict_crop_dt():
    data = request.json
    input_data = [data['state'], data['district'], data['season']]
    
    def classify(row, node):
        if isinstance(node, Leaf):
            return node.predictions
        if node.question.match(row):
            return classify(row, node.true_branch)
        else:
            return classify(row, node.false_branch)
    
    prediction = classify(input_data, dt_model)
    return jsonify({'predicted_crops': list(prediction.keys())})

if __name__ == '__main__':
    app.run(debug=True)