import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load cleaned dataset
df = pd.read_csv("cleaned_spam_dataset.csv")

# Check if necessary columns exist
if "message" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'message' and 'label' columns!")

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove punctuation and numbers
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Apply preprocessing to messages
df["message"] = df["message"].apply(preprocess_text)

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

