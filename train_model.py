import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("../dataset/resume.csv")

X = df["resume_text"]
y = df["role"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

model = nb_model if nb_acc > lr_acc else lr_model

print("NB Accuracy:", nb_acc)
print("LR Accuracy:", lr_acc)

pickle.dump(model, open("saved_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))