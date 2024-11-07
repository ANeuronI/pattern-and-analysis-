import numpy as np

class NaiveBayesClassifier:

    def __init__(self):
        """
        Initializes the classifier.
        """
        self.classes = None
        self.class_priors = {}
        self.feature_likelihoods = {}

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Parameters:
        X (numpy array): Feature values.
        y (numpy array): Class labels.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")

        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / len(X)
            self.feature_likelihoods[cls] = {}
            unique_features = np.unique(X)
            for feature_value in unique_features:
                self.feature_likelihoods[cls][feature_value] = (np.sum(X_cls == feature_value) + 1) / (len(X_cls) + len(unique_features))

    def predict(self, X):
        """
        Predicts class labels for new feature values.

        Parameters:
        X (numpy array): Feature values.

        Returns:
        predictions (list): Predicted class labels.
        """
        predictions = []
        for x in X:
            class_probabilities = {}
            for cls in self.classes:
                class_probabilities[cls] = self.class_priors[cls]
                if x in self.feature_likelihoods[cls]:
                    class_probabilities[cls] *= self.feature_likelihoods[cls][x]
                else:
                    # Handle unseen feature values
                    class_probabilities[cls] *= 1 / (len(self.feature_likelihoods[cls]) + len(np.unique(X)))
            prediction = max(class_probabilities, key=class_probabilities.get)
            print(f"Weather: {x}")
            print("Probabilities:")
            for cls, prob in class_probabilities.items():
                print(f"  {cls}: {prob:.4f}")
            print(f"Prediction: {prediction}\n")
            predictions.append(prediction)
        return predictions

# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array(['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'])
    y = np.array(['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'])

    # Initialize and train the classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X, y)

    # Predict new data
    X_test = np.array(['sunny', 'overcast', 'rainy'])
    predictions = nb_classifier.predict(X_test)



