from flask import Flask, render_template, request,app,jsonify

import pickle 

import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalaar = pickle.load(open('scalarmodel.pkl','rb'))

@app.route("/") 
def hello_world():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalaar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0]) 


if __name__ == "__main__":
    app.run(debug=True) 