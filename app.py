from flask import Flask, request, jsonify, render_template
import pickle
import numpy
from classifier import trainingModel

#initialise the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get_status", methods=["GET"])
def get_status():
    return {"training":70,"testing":30}

@app.route("/prediction", methods=["POST"])
def prediction():
    payload = request.json
    X_unknown = [
        payload["sepal-lenght"],
        payload["sepal-width"],
        payload["petal-lenght"],
        payload["petal-width"]
    ]
    X_unknown = numpy.array(X_unknown).reshape(1,-1)
    with open("./model/iris_classifier.pkl","rb") as f:
        clf = pickle.load(f)
    prediction = clf.predict(X_unknown)
    return jsonify({"predicted_value":prediction[0]})

@app.route("/training", methods=["POST"])
def training():
    trainingModel()
    return jsonify({"Model":"Model file created"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
