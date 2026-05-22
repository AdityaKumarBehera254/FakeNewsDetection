import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load CSV files
fake1 = pd.read_csv("fake 1.csv")
fake2 = pd.read_csv("fake 2.csv")

real1 = pd.read_csv("real 1.csv")
real2 = pd.read_csv("real 2.csv")

# Merge fake datasets
fake = pd.concat([fake1, fake2], axis=0)

# Merge real datasets
true = pd.concat([real1, real2], axis=0)

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge all data
data = pd.concat([fake, true], axis=0)

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Features and labels
X = data["text"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to vectors
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# User input
while True:
    news = input("\nEnter news text (or type exit): ")

    if news.lower() == "exit":
        break

    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)

    if prediction[0] == 1:
        print("Real News")
    else:
        print("Fake News")