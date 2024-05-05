# Data Preprocessing

The dataset is loaded from the CSV file using pandas.
The dataset is randomly shuffled using data.sample() with a fixed random state for reproducibility.
The features (X) and target variable (y) are separated. The target variable is the 'spam' column, indicating whether an email is spam or not.
The dataset is split into training (80%) and test (20%) sets using train_test_split() from scikit-learn.

# Naive Bayes

The NaiveBayes class is implemented from scratch.
In the fit() method, the class priors, feature means, and feature variances are calculated for each class.
The predict() method calculates the posterior probabilities for each class using the Gaussian probability density function and makes predictions based on the highest posterior probability.

# Logistic Regression

The LogisticRegression class is implemented from scratch.
The fit() method trains the logistic regression model using gradient descent. It updates the weights and bias iteratively based on the predicted probabilities and the actual labels.
The predict() method makes predictions using the trained weights and bias. It applies the sigmoid activation function to the linear predictor and classifies samples based on a threshold of 0.5.

# KNN

The KNN class is implemented from scratch.
The fit() method simply stores the training data.
The predict() method finds the k nearest neighbors for each sample in the test set using the Euclidean distance. It then makes predictions based on the majority vote of the labels of the k nearest neighbors.

# Cross-Validation

The cross_validation() function performs k-fold cross-validation on a given model, dataset, and number of folds (default is 5).
It splits the data into k folds, trains the model on k-1 folds, and evaluates it on the remaining fold. This process is repeated for each fold, and the average accuracy across all folds is calculated.

# Performance
The models are evaluated using cross-validation on the training set and then trained on the entire training set and evaluated on the test set.

## Cross-Validation Results

Naive Bayes Accuracy: 0.65

Logistic Regression Accuracy: 0.60

KNN Accuracy: 0.79

## Test Set Results

Naive Bayes Accuracy: 0.68

Logistic Regression Accuracy: 0.50

KNN Accuracy: .80
