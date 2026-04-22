import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df = pd.read_csv("../dataset/resume.csv")

X = df["resume_text"]
y = df["role"]

vectorizer = pickle.load(open("../model/vectorizer.pkl", "rb"))
X_vec = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC()
}

accuracies = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    print(f"{name}: {acc}")

os.makedirs("../static/plots", exist_ok=True)

best_model = MultinomialNB()
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.title("Confusion Matrix")
plt.savefig("../static/plots/confusion_matrix.png")
plt.close()

plt.figure()
plt.plot(list(models.keys()), accuracies, marker='o')
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.savefig("../static/plots/accuracy.png")
plt.close()

print("Graphs generated successfully!")
