from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('diabetes_model123.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        return render_template('home.html', prediction_text= 'Yahoooooo! You will not suffer Diabetes :)')
    else:
        return render_template('home.html', prediction_text= 'Alas! Be careful. You might suffer from Diabetes :(')


if __name__ == "__main__":
    app.run(debug=True)
