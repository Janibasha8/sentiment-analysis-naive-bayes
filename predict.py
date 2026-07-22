import joblib

model=joblib.load("../models/sentiment_model.pkl")

while True:

    review=input("Enter Review: ")

    prediction=model.predict([review])[0]

    print(prediction)
