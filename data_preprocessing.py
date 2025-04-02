import pandas as pd

# Load dataset
df = pd.read_csv("spam (1).csv", encoding="latin-1")

# Display column names to identify relevant ones
print("Columns in dataset:", df.columns)

# Select only the relevant columns (assuming 'v1' is label and 'v2' is message)
df = df.iloc[:, [0, 1]]  # Keep only first two columns

# Rename columns properly
df.columns = ['label', 'message']

# Drop missing values in essential columns
df = df.dropna(subset=['label', 'message'])

# Convert labels to binary format (spam = 1, ham = 0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Save the cleaned dataset
df.to_csv("cleaned_spam_dataset.csv", index=False)

print("✅ Data preprocessing completed. Cleaned dataset saved as 'cleaned_spam_dataset.csv'.")



