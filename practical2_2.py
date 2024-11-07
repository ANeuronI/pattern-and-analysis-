from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Sample data
texts = [
    "I love programming in Python",
    "Python is a great language",
    "I dislike bugs in the code",
    "Debugging is fun",
    "I enjoy learning new algorithms",
    "Machine learning is fascinating",
    "I hate syntax errors",
    "Coding challenges are exciting",
    "Programming is my passion",
    "I'm frustrated with this code",
    "This library is amazing",
    "I'm stuck on this problem",
    "The documentation is terrible",
    "I love solving puzzles",
    "This tutorial is helpful",
    "The community is supportive",
    "I'm annoyed with this error",
    "The code is so elegant",
    "I'm excited to learn more",
    "This project is challenging"
]

labels = [
    "positive", "positive", "negative", "positive", "positive", "positive", "negative", "positive",
    "positive", "negative", "positive", "negative", "negative", "positive", "positive", "positive",
    "negative", "positive", "positive", "positive"
]

# Balance the dataset
negative_texts = [text for text, label in zip(texts, labels) if label == 'negative']
positive_texts = [text for text, label in zip(texts, labels) if label == 'positive']

# Add more negative samples
negative_texts += negative_texts  # Duplicate negative samples

# Combine balanced texts and labels
balanced_texts = negative_texts + positive_texts
balanced_labels = ['negative'] * len(negative_texts) + ['positive'] * len(positive_texts)

# Shuffle the balanced data
balanced_texts, balanced_labels = shuffle(balanced_texts, balanced_labels, random_state=None)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(balanced_texts, balanced_labels, test_size=0.25)

# Create a pipeline that combines the TF-IDF vectorizer and the Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict the labels for the test set
predictions = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)

# Perform cross-validation
cv_scores = cross_val_score(model, texts, labels, cv=5)

print("\n Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", cv_scores.mean())
print("\n Predictions: \n \n", predictions)
print("\n Accuracy:", accuracy)
print("\n")
# Function to predict sentiment and probabilities

def predict_sentiment(sentence):
    probabilities = model.predict_proba([sentence])[0]
    prediction = model.predict([sentence])[0]
    print(f"\nSentence: {sentence}")
    print(f"Prediction: {prediction}")
    print(f"Probabilities: Positive: {probabilities[1]:.2f}, Negative: {probabilities[0]:.2f}")
    
# Example usage
predict_sentiment("I love this new feature")


