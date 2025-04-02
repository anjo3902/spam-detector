import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Load cleaned dataset
df = pd.read_csv("cleaned_spam_dataset.csv")

# Check if necessary columns exist
if "message" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'message' and 'label' columns!")

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["message"])  # Use "message" column
y = df["label"]

# Split data into training and test set (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Train Naïve Bayes Model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

# Train Decision Tree Model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

# Save Models & Vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("logistic_model.pkl", "wb") as f:
    pickle.dump(logistic_model, f)

with open("naive_bayes_model.pkl", "wb") as f:
    pickle.dump(naive_bayes_model, f)

with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(decision_tree_model, f)

print("Models trained and saved successfully!")

