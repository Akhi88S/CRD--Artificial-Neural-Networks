import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model12 = pickle.load(open('model12.pkl', 'rb'))
model = pickle.load(open('KNN.pkl', 'rb'))
model1 = pickle.load(open('SVC.pkl', 'rb'))
model2 = pickle.load(open('LogisticRegresion.pkl', 'rb'))
model3 = pickle.load(open('LinearAnalysis.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    prediction111 = model12.predict(final_features)
    
    prediction = model.predict(final_features)  
    
    prediction1 = model1.predict(final_features)
    
    prediction2 = model2.predict(final_features)
    
    prediction3 = model3.predict(final_features)
    
    output111 = prediction111
    
    output = prediction
    
    output1 = prediction1
    
    output2 = prediction2
    
    output3 = prediction3
    
    print("ANN Without Hidden  ",output111)
    print()
    
    print("KNN  ",output)
    print()
    
    print("SVC  ",output1)
    print()
    
    print("Logistic Regression  ",output2)
    print()
    
    
    print("Linear DIscriminant Analysis  ",output3)
    print()
    
    if(output111 >= 1):
        return render_template('index.html', prediction_text='You Dont have CRD')
    elif(output111 < 1):
        return render_template('index.html', prediction_text='You have CRD ')
        
        
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)