import glob
import tensorflow as tf
from keras.layers import Dense
from keras import Sequential
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras import initializers
import keras as k # in case I miss any methods I want to use
from keras.layers import Dropout
import pickle
import os

here = os.path.dirname(os.path.abspath(__file__))
# from google.colab import files #Only use for Google Colab
# uploaded = files.upload()      #Only use for Google Colab
df = pd.read_csv("kidney_disease.csv")

#Create a list of columns to retain
columns_to_retain = ["sg", "al", "sc", "hemo",
                         "pcv", "wbcc", "rbcc", "htn", "classification"]

#columns_to_retain = df.columns, Drop the columns that are not in columns_to_retain
df = df.drop([col for col in df.columns if not col in columns_to_retain], axis=1)
    
# Drop the rows with na or missing values
df = df.dropna(axis=0)

#Transform non-numeric columns into numerical columns
for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])

#Print / show the first 5 rows of the new cleaned data set

X = df.drop(["classification"], axis=1)
y = df["classification"]
#########z=df.head()
############print(z)
#Feature Scaling
#the min-max scaler method scales the dataset so that all the input features lie between 0 and 1 inclusive

############################################################################################################

X_train,  X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.4, shuffle=True,random_state=42)


model123 = Sequential()
model123.add(Dense(units = 256, input_dim=len(X.columns), kernel_initializer ='uniform', activation = 'relu'))

#model123.add(Dense(units = 128, kernel_initializer ='uniform', activation = 'relu')) # HIDDEN LAYER
#model123.add(Dropout(0.2))

model123.add(Dense(1, activation="hard_sigmoid"))
# Compiling the ANN
model123.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Train the model
mode123 = model123.fit(X_train, y_train, 
                    epochs=10 #The number of iterations over the entire dataset to train on
                    ) #The number of samples per gradient update for training
print()
print()

model123.save("ckd.model")
#Visualize the models accuracy and loss
#X_train=np.array(X_train)
#y_train=np.array(y_train)
#X_test=np.array(X_test)
#Loop through any and all saved models. Then get each models accuracy, loss, prediction and original values on the test data.
for model_file in glob.glob("*.model"):
  model123 = load_model(model_file)
  pred = model123.predict(X_test)   #([[[11111111111, 0  , 0 , 1111111113 ,  33   , 0]]])
  pred = [1 if y>=0.5 else 0 for y in pred]
  scores = model123.evaluate(X_test, y_test)
  print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
  print()
  print("Predicted : {0}".format(", ".join([str(x) for x in pred])))
  print()
  print("**Accuracy of ANN with out hidden layer**",scores[1]*100)
  pickle.dump(model123, open('model111.pkl','wb'))     
  model123 = pickle.load(open('model111.pkl','rb'))

print()
print()

print("* "*84)

#input_shape=(X_train.shape[1],)

X_train,  X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.4, shuffle=True,random_state=1)


model111 = Sequential()
model111.add(Dense(units = 256, input_dim=len(X.columns), kernel_initializer ='uniform', activation = 'relu'))

model111.add(Dense(units = 128, kernel_initializer ='uniform', activation = 'relu')) # HIDDEN LAYER
model111.add(Dropout(0.2))

model111.add(Dense(1, activation="hard_sigmoid"))
# Compiling the ANN
model111.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Train the model
model111 = model111.fit(X_train, y_train, 
                    epochs=10 #The number of iterations over the entire dataset to train on
                    ) #The number of samples per gradient update for training

print()
print()

#Visualize the models accuracy and loss
#X_train=np.array(X_train)
#y_train=np.array(y_train)
#X_test=np.array(X_test)
#Loop through any and all saved models. Then get each models accuracy, loss, prediction and original values on the test data.
for model_file in glob.glob("*.model"):
  model111 = load_model(model_file)
  pred = model111.predict(X_test)   #([[[11111111111, 0  , 0 , 1111111113 ,  33   , 0]]])
  pred = [1 if y>=0.5 else 0 for y in pred]
  scores = model111.evaluate(X_test, y_test)
  print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
  print()
  print("Predicted : {0}".format(", ".join([str(x) for x in pred])))
  print()
  print("**Accuracy of ANN with hidden layer**",scores[1]*100)
  pickle.dump(model111, open('model111.pkl','wb'))     
  model111 = pickle.load(open('model111.pkl','rb'))


print()
print()
print("* "*84)


from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, f1_score, log_loss
from sklearn.metrics import classification_report,confusion_matrix, precision_recall_fscore_support 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


X_train,  X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.4, shuffle=True,random_state=42)
classifiers = [
KNeighborsClassifier(n_neighbors=35, metric='euclidean', p=2),
      ]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", 'Log Loss']
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    try:
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
        print()
        print("Predicted : {0}".format(", ".join([str(x) for x in train_predictions])))
        print()
        print("**Accuracy of KNN Classifier**",acc*100)
        
        pickle.dump(clf, open('KNN.pkl','wb'))     
        model = pickle.load(open('KNN.pkl','rb'))
        
    except Exception as e:
        print (e)

print()
print()
print("* "*84)



classifiers1 = [
     SVC(C=.1, degree=1, kernel='poly', probability=True),
      ]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", 'Log Loss']
log = pd.DataFrame(columns=log_cols)

for clf in classifiers1:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    try:
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
        print()
        print("Predicted : {0}".format(", ".join([str(x) for x in train_predictions])))
        print()
        print("**Accuracy of SVC**",acc*100)
 
        pickle.dump(clf, open('SVC.pkl','wb'))     
        model1 = pickle.load(open('SVC.pkl','rb'))
        
    except Exception as e:
        print (e)


print()
print()
print("* "*84)


classifiers2 = [
    LogisticRegression(penalty = 'l2',C=0.1,random_state=0),
      ]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", 'Log Loss']
log = pd.DataFrame(columns=log_cols)

for clf in classifiers2:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    try:
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
        print()
        print("Predicted : {0}".format(", ".join([str(x) for x in train_predictions])))
        print()
        print("**Accuracy of LogisticRegression Classifier**",acc*100)
 
        pickle.dump(clf, open('LogisticRegresion.pkl','wb'))     
        model2 = pickle.load(open('LogisticRegresion.pkl','rb'))
        
    except Exception as e:
        print (e)


print()
print()
print("* "*84)


classifiers3 = [
    LinearDiscriminantAnalysis(), 
      ]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", 'Log Loss']
log = pd.DataFrame(columns=log_cols)

for clf in classifiers3:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    try:
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
        print()
        print("Predicted : {0}".format(", ".join([str(x) for x in train_predictions])))
        print()
        print("**Accuracy of LinearDiscriminantAnalysis**",acc*100)
 
        pickle.dump(clf, open('LinearAnalysis.pkl','wb'))     
        model3 = pickle.load(open('LinearAnalysis.pkl','rb'))
        
    except Exception as e:
        print (e)


#print(model.predict([[1.020, 1.0, 1.2,15,28,1]]))
