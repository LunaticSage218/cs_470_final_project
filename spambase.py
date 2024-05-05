# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from the CSV file
data = pd.read_csv('C:\\Users\\Larry\\Code\\Python\\CS470\\final_project\\spambase.csv')

data = data.sample(frac=1, random_state=42)

# set X and y
X = data.drop(['spam'], axis=1).values
y = data['spam'].values

# Split the dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes implementation
class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.feature_means = None
        self.feature_vars = None

    def fit(self, X, y):
        # Calculate class priors, feature means, and feature variances for each class
        self.classes = np.unique(y)
        self.class_priors = np.array([np.mean(y == c) for c in self.classes])
        self.feature_means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.feature_vars = np.array([X[y == c].var(axis=0) for c in self.classes])

    def predict(self, X):
        # Calculate posterior probabilities for each class and make predictions
        posteriors = []
        eps = 1e-9  # Small epsilon value to avoid division by zero
        for c in self.classes:
            prior = np.log(self.class_priors[c])
            conditional = np.sum(np.log(self.gaussian_pdf(X, self.feature_means[c], self.feature_vars[c]) + eps), axis=1)
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors, axis=0)]

    def gaussian_pdf(self, X, mean, var):
        # Calculate the Gaussian probability density function for each feature
        eps = 1e-9  # Small epsilon value to avoid division by zero
        const = 1 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-((X - mean) ** 2) / (2 * (var + eps)))
        return const * exponent

# Logistic Regression implementation
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Train the logistic regression model using gradient descent
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Make predictions using the trained logistic regression model
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        class_pred = [1 if i > 0.5 else 0 for i in y_pred]
        return class_pred

    def sigmoid(self, x):
        # Apply the sigmoid activation function
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

# KNN implementation
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        # Store the training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Make predictions for each sample in X
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Find the k nearest neighbors and make a prediction based on majority vote
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in indices]
        most_common = np.argmax(np.bincount(k_nearest_labels))
        return most_common

    def euclidean_distance(self, x1, x2):
        # Calculate the Euclidean distance between two samples
        return np.sqrt(np.sum((x1 - x2) ** 2))

# 5-fold cross-validation
def cross_validation(model, X, y, k=5):
    fold_size = len(X) // k
    scores = []

    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size

        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.concatenate((X[:val_start], X[val_end:]), axis=0)
        y_train = np.concatenate((y[:val_start], y[val_end:]), axis=0)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        scores.append(accuracy)

    return np.mean(scores)

# Evaluate models using cross-validation
nb_model = NaiveBayes()
lr_model = LogisticRegression()
knn_model = KNN()

nb_scores = cross_validation(nb_model, X_train, y_train)
lr_scores = cross_validation(lr_model, X_train, y_train)
knn_scores = cross_validation(knn_model, X_train, y_train)

print("Naive Bayes Accuracy: {:.2f}".format(nb_scores))
print("Logistic Regression Accuracy: {:.2f}".format(lr_scores))
print("KNN Accuracy: {:.2f}".format(knn_scores))

# Train on the entire training set and evaluate on the test set
nb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

nb_accuracy = np.mean(nb_pred == y_test)
lr_accuracy = np.mean(lr_pred == y_test)
knn_accuracy = np.mean(knn_pred == y_test)

print("\nTest Accuracy:")
print("Naive Bayes: {:.2f}".format(nb_accuracy))
print("Logistic Regression: {:.2f}".format(lr_accuracy))
print("KNN: {:.2f}".format(knn_accuracy))