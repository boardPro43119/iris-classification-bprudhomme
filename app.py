import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    class_num = round(prediction[0])
    if class_num == 0:
        output = "Iris-setosa"
    elif class_num == 1:
        output = "Iris-versicolor"
    elif class_num == 2:
        output = "Iris-virginica"
    return render_template('index.html', prediction_text='Predicted species: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)