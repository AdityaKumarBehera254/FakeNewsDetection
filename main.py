import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load CSV files
fake = pd.read_csv("Fake_small.csv")
true = pd.read_csv("True_small.csv")

# Add labels
fake["label"] = 0     # Fake news
true["label"] = 1     # True news

# Merge datasets
data = pd.concat([fake, true], axis=0)

# Shuffle data
data = data.sample(frac=1, random_state=42)

# Keep only text and label columns
data = data[["text", "label"]]

# Split data
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))

# User input prediction
while True:
    news = input("\nEnter news text (or type exit): ")

    if news.lower() == "exit":
        break

    news_vector = vectorizer.transform([news])

    result = model.predict(news_vector)

    if result[0] == 0:
        print("Prediction: Fake News")
    else:
        print("Prediction: True News")
