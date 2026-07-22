import joblib

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from data_preprocessing import load_data

df = load_data("../data/IMDB Dataset.csv")

X_train,X_test,y_train,y_test=train_test_split(

    df["review"],
    df["sentiment"],
    test_size=0.20,
    random_state=42

)

model=joblib.load("../models/sentiment_model.pkl")

pred=model.predict(X_test)

print(classification_report(y_test,pred))

print("Accuracy:",accuracy_score(y_test,pred))

cm=confusion_matrix(y_test,pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("../images/confusion_matrix.png")
