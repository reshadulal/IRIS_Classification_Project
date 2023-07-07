import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib ##for saving model

#random seed
seed = 42

#read original dataset
# iris_df = pd.read_csv("C:/Users/DELL/Desktop/Iris_Project/IRIS_Classification_Project/data/Iris.csv")
# iris_df = pd.read_csv("C:\\Users\\DELL\\Desktop\\Iris_Project\\IRIS_Classification_Project\\data/Iris.csv")
# iris_df = pd.read_csv("C:\\Users\DELL\Desktop\Iris_Project\IRIS_Classification_Project\data/Iris.csv")
iris_df = pd.read_csv("data\Iris.csv")

#selecting features and target data
X = iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris_df[['Species']]

#split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=seed, stratify=y)

#create an instance of the k neighbour classifier
clf = KNeighborsClassifier(n_neighbors=10)

#train the classifier on the training data
# clf.fit(X_train,np.ravel(y_train,order='C'))
clf.fit(X_train.values,y_train.values.ravel())

#predict on the test set
y_pred = clf.predict(X_test.values)

#calculate accuracy
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy}") #accuracy :0.91

#save the model to disk
joblib.dump(clf,"output_models/kn_model.sav")