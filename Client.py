from flask import Flask, render_template, request
import pandas as pd
import pickle
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        # Get the inputs from the form
        inputs = request.form
        input_df = pd.DataFrame(inputs, index=[0])
        input_df=input_df.mask(input_df == '')
        input_df = input_df.apply(pd.to_numeric, errors='ignore')
        # Convert the inputs to the correct data types
        input_df = input_df.astype({'age': int, 'sex': int, 'cp': int, 'trtbps': int,
                                    'restecg': int, 'thalachh': int,
                                    'exng': int, 'oldpeak': float, 'slp': int, 'caa': int,
                                    'thall': int}, errors='ignore')
        # Load the trained logistic regression model
        model = pickle.load(open('./model/LogisticRegression.sav', 'rb'))
        # Use the model to make predictions
        predictions = model.predict(input_df)
        print(predictions)
        # Return the prediction results to the template
        return render_template('index.html', prediction=predictions[0] + 1) # predictions[0]
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)