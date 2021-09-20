from flask import Flask, render_template , request
import joblib


# importing all the important libraires
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# download the model
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

# initialse the app
app = Flask(__name__)

#load the model
tfidf = joblib.load('tfidf_vector_model.pkl')
model = joblib.load('netflix_75.pkl')

@app.route('/')
def hello():
    return render_template('form.html')

@app.route('/submit' , methods = ["POST"])
def form_data():
     user_data = request.form.get('user_data')
     user_data1 = [user_data]
     vector = tfidf.transform(user_data1)
     my_pred = model.predict(vector)

     if my_pred[0] == 1:
          out = 'positve review'
     else:
          out = 'negative review'
  
   

     

     return render_template('predict.html' , data = f' {out}')

if __name__ == '__main__':
    app.run(debug = True)

