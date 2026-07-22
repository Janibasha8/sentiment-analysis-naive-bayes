import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from data_preprocessing import load_data

df = load_data("../data/IMDB Dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(

    df["review"],
    df["sentiment"],
    test_size=0.20,
    random_state=42

)

pipeline = Pipeline([

    ("vectorizer", CountVectorizer()),

    ("classifier", MultinomialNB())

])

pipeline.fit(X_train,y_train)

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test,predictions)

print("Accuracy:",accuracy)

joblib.dump(pipeline,"../models/sentiment_model.pkl")
