import joblib

def predict(data):
    clf = joblib.load("output_models/kn_model.sav")
    return clf.predict(data)