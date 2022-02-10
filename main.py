from flask import Flask,render_template,url_for,request
import pickle
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
classifier = pickle.load(open('classifier.pkl','rb'))

app  = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = 'POST')
def predict():
    if request.method == 'POST':
        user_message  = request.form['message']
        data = [user_message]
        input_vector = vectorizer.tranform(data).toarray()
        my_prediction = classifier.predict(input_vector)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(port = 5001,debug=True)