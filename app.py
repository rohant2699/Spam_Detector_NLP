from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf=pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        vect1=pd.DataFrame(vect)
        my_prediction=clf.predict(vect1)

    return render_template('result.html',prediction=my_prediction)

if  __name__ == "__main__":
    app.run(debug=True)