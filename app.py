
import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
#app = Flask(__name__)

#Load the trained model. (Pickle file)
flask_app = Flask(__name__ ,template_folder="template")
model = pickle.load(open('model.pkl', 'rb'))



#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@flask_app.route('/')
def home():
    return render_template('index.html')

#Redirect to /predict page with the output
@flask_app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Percent with heart disease is {}'.format(output))


if __name__ == "__main__":
    flask_app.run()