import pandas as pd

# Machine Learning tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------
# STEP 1: Load datasets
# -----------------------------------

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# -----------------------------------
# STEP 2: Add labels
# Fake = 0
# Real = 1
# -----------------------------------

fake["label"] = 0
true["label"] = 1

# -----------------------------------
# STEP 3: Combine datasets
# -----------------------------------

data = pd.concat([fake, true])

# Keep only text and label
data = data[["text", "label"]]

# -----------------------------------
# STEP 4: Input and Output
# -----------------------------------

X = data["text"]
y = data["label"]

# -----------------------------------
# STEP 5: Split training and testing
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# STEP 6: Convert text into numbers
# -----------------------------------

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# -----------------------------------
# STEP 7: Train AI model
# -----------------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------------
# STEP 8: Test model accuracy
# -----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------------
# STEP 9: User input prediction
# -----------------------------------

news = input("\nEnter news text:\n")

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

# -----------------------------------
# STEP 10: Show result
# -----------------------------------

if prediction[0] == 0:
    print("\n🟥 Fake News")
else:
    print("\n🟩 Real News")
