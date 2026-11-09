import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

load_dotenv()

test_size = float(os.getenv("TEST_SIZE", "0.2"))
random_seed = int(os.getenv("RANDOM_SEED", "42"))

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
print(df.head())
print(df.shape)
print(df.describe())

X = df[iris.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

with open("wyniki.txt", "w", encoding="utf-8") as f:
    f.write(f"accuracy_score: {acc}\n\n{report}")

print(acc)
print(report)
