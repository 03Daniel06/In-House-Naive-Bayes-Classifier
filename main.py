"""
Hi! I'm Daniel Northcott, an engineering student with minors in Computer Science and Mathematics, set to graduate in December 2025. 
This project is part of my ongoing efforts to build a deeper understanding of machine learning algorithms, 
and here I've implemented a Naive Bayes classifier from scratch. The goal is to use this classifier to predict classes 
in a dataset and evaluate its performance. The project also includes data visualization and performance metrics 
to analyze classification results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set a random seed for reproducibility
np.random.seed(42)

class NaiveBayesClassifier:
    def __init__(self, num_bins=10):
        """
        Initialize the Naive Bayes classifier with specified number of bins.
        
        Parameters:
        num_bins: Number of bins to discretize feature values into.
        """
        self.class_priors = {}  # Prior probabilities of each class
        self.feature_likelihoods = {}  # Likelihoods for each feature per class
        self.classes = []  # List of unique classes
        self.num_bins = num_bins  # Number of bins for discretizing feature values
        self.bins = np.linspace(0, 1, num_bins + 1)  # Bin edges for features in range [0, 1]

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier by calculating prior probabilities and likelihoods.
        
        Parameters:
        X: Training feature matrix.
        y: Training labels.
        """
        self.classes = np.unique(y)  # Identify unique class labels
        num_samples, num_features = X.shape  # Get the shape of the feature matrix
        
        # Calculate prior and likelihoods for each class
        for c in self.classes:
            # Calculate the prior probability P(c)
            X_c = X[y == c]  # Subset of X where class is c
            self.class_priors[c] = X_c.shape[0] / float(num_samples)  # P(c)
            
            # Calculate the likelihood P(xd | c) for each feature
            likelihoods = {}
            for i in range(num_features):
                feature_vals = X_c[:, i]  # Get values for the i-th feature
                # Create a histogram for the feature, count occurrences in bins
                bin_counts, _ = np.histogram(feature_vals, bins=self.bins)
                likelihoods[i] = bin_counts / X_c.shape[0]  # Normalized bin counts
            self.feature_likelihoods[c] = likelihoods  # Store the likelihoods for class c

    def _calculate_likelihood(self, bin_probs, x):
        """
        Calculate likelihood based on the bin probability for the given feature.
        
        Parameters:
        bin_probs: Array of probabilities for each bin.
        x: Feature value to calculate likelihood for.
        
        Returns:
        The probability of x given the bin distribution.
        """
        bin_index = np.digitize(x, self.bins) - 1  # Find bin index for x
        bin_index = min(max(bin_index, 0), self.num_bins - 1)  # Ensure bin index is within bounds
        return bin_probs[bin_index]  # Return the likelihood

    def predict(self, X):
        """
        Predict the class for each sample in X.
        
        Parameters:
        X: Test feature matrix.
        
        Returns:
        Predicted class labels for each sample.
        """
        y_pred = []
        for x in X:
            class_probabilities = {}  # Dictionary to store class probabilities
            for c in self.classes:
                prior = np.log(self.class_priors[c])  # Log of prior probability
                likelihood = 0
                for i, feature_value in enumerate(x):
                    bin_probs = self.feature_likelihoods[c][i]  # Get bin probabilities for class c
                    likelihood_value = self._calculate_likelihood(bin_probs, feature_value)
                    likelihood += np.log(likelihood_value + 1e-10)  # Avoid log(0)
                class_probabilities[c] = prior + likelihood  # Combine prior and likelihood
            y_pred.append(max(class_probabilities, key=class_probabilities.get))  # Select class with max probability
        return np.array(y_pred)

# Function to plot 2D slices of the 8D data
def plot_2d_slice(X, y, column_x, column_y, dataset_name, set_number):
    """
    Plot a 2D slice of the data based on two selected columns.
    
    Args:
        X: Feature matrix.
        y: Target labels.
        column_x: Index of the feature for the x-axis.
        column_y: Index of the feature for the y-axis.
        dataset_name: Name of the dataset for plot title.
        set_number: Dataset number for plot title.
    """
    plt.figure(figsize=(8, 6))
    classes = np.unique(y)  # Get unique classes
    
    for c in classes:
        plt.scatter(X[y == c, column_x], X[y == c, column_y], label=f"Class {c}")
    
    plt.xlabel(f"Feature {column_x}")
    plt.ylabel(f"Feature {column_y}")
    plt.legend()
    plt.title(f"2D Slice: Feature {column_x} vs Feature {column_y} ({dataset_name} Set {set_number})")
    plt.grid(True)
    plt.show()

# Example usage:

def read_data(file_path):
    """Reads data from a file and stores it in a matrix."""
    return np.loadtxt(file_path)

def read_labels(file_path):
    """Reads labels from a file and stores them in a vector."""
    return np.loadtxt(file_path).astype(np.int64)

# Function to prompt user to select a dataset
def get_user_choice():
    """
    Prompts the user to select which dataset to use for training and testing.
    
    Returns:
    The dataset number selected by the user as a string.
    """
    while True:
        choice = input("Please enter the dataset number you want to use (1-4): ")
        if choice in ['1', '2', '3', '4']:  # Validate input
            return choice
        else:
            print("Invalid input. Please enter a number between 1 and 4.")
def main():
    # Get the user's choice of dataset
    dataset_number = get_user_choice()

    # Construct file names for training and testing based on the user's choice
    data_train_file = f'Data-{dataset_number}-train.txt'
    label_train_file = f'Label-{dataset_number}-train.txt'
    data_test_file = f'Data-{dataset_number}-test.txt'
    label_test_file = f'Label-{dataset_number}-test.txt'

    # Load the training and testing datasets
    X_train = read_data(data_train_file)  # Load training feature matrix
    y_train = read_labels(label_train_file)  # Load training labels
    X_test = read_data(data_test_file)  # Load testing feature matrix
    y_test = read_labels(label_test_file)  # Load testing labels

    # Initialize and train Naive Bayes Classifier with 10 bins
    nb_classifier = NaiveBayesClassifier(num_bins=10)
    nb_classifier.fit(X_train, y_train)

    # Predict on test data
    y_pred = nb_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Total Accuracy: {accuracy * 100:.4f}%")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,digits=4))

    # Print accuracy for each class
    for class_label, metrics in class_report.items():
        if class_label.isdigit():  # Check if the key is a class label
            print(f"Accuracy for class {class_label}: {metrics['precision']:.6f}")

if __name__ == "__main__":
    main()
