import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


np.random.seed(42)

class NaiveBayesClassifier:
    def __init__(self, num_bins=10):
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.classes = []
        self.num_bins = num_bins
        self.bins = np.linspace(0, 1, num_bins + 1)  # Bin edges for features in range [0, 1]

    def fit(self, X, y):
        """Train the Naive Bayes classifier."""
        self.classes = np.unique(y)
        num_samples, num_features = X.shape
        
        for c in self.classes:
            # Calculate the prior probability P(c)
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / float(num_samples)
            
            # Calculate the likelihood P(xd | c) for each feature
            likelihoods = {}
            for i in range(num_features):
                feature_vals = X_c[:, i]
                # Create a histogram for the feature, count occurrences in bins
                bin_counts, _ = np.histogram(feature_vals, bins=self.bins)
                # Store the probabilities (normalized bin counts)
                likelihoods[i] = bin_counts / X_c.shape[0]
            self.feature_likelihoods[c] = likelihoods

    def _calculate_likelihood(self, bin_probs, x):
        """Calculate likelihood based on the bin probability for the given feature."""
        # Find the bin index that x falls into
        bin_index = np.digitize(x, self.bins) - 1  # -1 to convert to 0-indexed
        bin_index = min(max(bin_index, 0), self.num_bins - 1)  # Ensure it's within range
        return bin_probs[bin_index]

    def predict(self, X):
        """Predict the class for each sample in X."""
        y_pred = []
        for x in X:
            class_probabilities = {}
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                likelihood = 0
                for i, feature_value in enumerate(x):
                    bin_probs = self.feature_likelihoods[c][i]
                    # Add a small constant to avoid log(0)
                    likelihood_value = self._calculate_likelihood(bin_probs, feature_value)
                    likelihood += np.log(likelihood_value + 1e-10)
                class_probabilities[c] = prior + likelihood
            y_pred.append(max(class_probabilities, key=class_probabilities.get))
        return np.array(y_pred)

# Function to plot 2D slices of the 8D data
def plot_2d_slice(X, y, column_x, column_y, dataset_name, set_number):
    """Plot a 2D slice of the data based on two selected columns.
    
    Args:
        X: Feature matrix (2D array).
        y: Target labels (1D array).
        column_x: Index of the feature for the x-axis.
        column_y: Index of the feature for the y-axis.
        dataset_name: Name of the dataset to include in the title.
        set_number: Number of the dataset to include in the title.
    """
    plt.figure(figsize=(8, 6))
    classes = np.unique(y)
    
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

def main():
     # File paths
    data_file_train = 'Data-3-train.txt'
    labels_file_train = 'Label-3-train.txt'
    data_file_test = 'Data-3-test.txt'
    labels_file_test = 'Label-3-test.txt'

    # Load the training data
    X_train = read_data(data_file_train)
    y_train = read_labels(labels_file_train)

    # Load the test data
    X_test = read_data(data_file_test)
    y_test = read_labels(labels_file_test)

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
