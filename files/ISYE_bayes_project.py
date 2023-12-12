"""
data set: IMDB Dataset of 50K Movie Reviews
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
"""


import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import math
import nltk
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')

class NaiveBayesClassifier:
    def __init__(self):
        self.w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.vectorizer = CountVectorizer(max_features=3000)
        self.labels = [0, 1]
        self.n_label_items = {}
        self.log_label_priors = {}
        self.word_counts = {}
        self.vocab = []

    # Remove HTML tags, URLs, non-alphanumeric characters and convert to lowercase
    def remove_tags(self, string):
        result = re.sub(r'<.*?>', '', string) 
        result = re.sub(r'https://.*', '', result) 
        result = re.sub(r'[^a-zA-Z\s]', ' ', result)
        result = result.lower()
        return result

    # Lemmatize text; finding the root form of the word
    def lemmatize_text(self, text):
        lemmatized_words = [self.lemmatizer.lemmatize(w) for w in self.w_tokenizer.tokenize(text)]
        return ' '.join(lemmatized_words)

    # Applies the above two functions to the data 'review' column
    def preprocess_data(self, data):
        data['review'] = data['review'].apply(self.remove_tags)
        stop_words = set(stopwords.words('english'))
        data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        data['review'] = data['review'].apply(self.lemmatize_text)
        return data

    # Calculate the log of the prior probability of each label
    def fit(self, x, y):
        n = len(x)
        grouped_data = self.group_by_label(x, y)
        for label, data in grouped_data.items():
            self.n_label_items[label] = len(data)
            if self.n_label_items[label] > 0:  # Check if the denominator is non-zero
                self.log_label_priors[label] = math.log(self.n_label_items[label] / n)
            else:
                self.log_label_priors[label] = float('-inf')  # Set to negative infinity for zero probability

    # Group the data by label
    def group_by_label(self, x, y):
        data = {label: x[np.where(y == label)] for label in self.labels}
        return data

    # Apply Laplace smoothing to a word given a label
    def laplace_smoothing(self, word, label):
        count_word_label = self.word_counts[label].get(word, 0) + 1
        total_label_items = self.n_label_items[label] + len(self.vocab)
        probability = count_word_label / total_label_items
        return math.log(probability)

    # Fit the word counts for each label
    def fit_word_counts(self, x, y):
        X = self.vectorizer.fit_transform(x)
        self.vocab = self.vectorizer.get_feature_names_out()
        X = X.toarray()
        for label in self.labels:
            self.word_counts[label] = defaultdict(lambda: 0)
        for i, label in enumerate(y):
            for j, word_count in enumerate(X[i]):
                self.word_counts[label][self.vocab[j]] += word_count

    # Implements the Naive Bayes classification logic by calculating the probability of each label
    # given the words in the input text and selecting the label with the highest probability as the prediction.
    def predict(self, input_texts):
        predictions = []

        for text in input_texts:
            label_scores = {label: self.log_label_priors[label] for label in self.labels}
            words = set(self.w_tokenizer.tokenize(text))

            for w in words:
                if w in self.vocab:
                    for l in self.labels:
                        log_w_given_l = self.laplace_smoothing(w, l)
                        label_scores[l] += log_w_given_l

            predictions.append(max(label_scores, key=label_scores.get))

        return predictions




# Take user input and make predictions on the fly
def predict_user_input(classifier, encoder):
    while True:
        user_input = input("Enter a movie review (type 'exit' to stop): ")
        if user_input.lower() == 'exit':
            break
        # Preprocess the user input
        user_input = classifier.preprocess_data(pd.DataFrame({'review': [user_input]}))['review'][0]
        # Make prediction
        pred = classifier.predict([user_input])
        # Convert predicted label back to original format
        pred_label = encoder.inverse_transform(pred)[0]
        print("Predicted sentiment:", pred_label)
        print()


# Load data
data = pd.read_csv('IMDB_Dataset.csv')
# print(data.head())


# Instantiate the classifier
classifier = NaiveBayesClassifier()

# Preprocess data
data = classifier.preprocess_data(data)

# Split data
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    data['review'].values, data['sentiment'].values, stratify=data['sentiment'].values
)

# Encode labels (0: positive, 1: negative)
encoder = LabelEncoder()
train_labels_encoded = encoder.fit_transform(train_labels)

# Fit the classifier
classifier.fit(train_sentences, train_labels_encoded)
classifier.fit_word_counts(train_sentences, train_labels_encoded)

# Make predictions on the test set
pred = classifier.predict(test_sentences)

# Convert predicted labels back to original format
pred_labels = encoder.inverse_transform(pred)

# Evaluate the model
print("Accuracy of prediction on the test set:", accuracy_score(test_labels, pred_labels))

# Predict sentiment for user input in a loop
predict_user_input(classifier, encoder)


