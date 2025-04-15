import numpy as np 


class UserMoodPredictionModel:
    def __init__(self, learning_rate=0.005, n_iterations=30000, regularization_strength=0.05):
        self.learning_rate = learning_rate  # Learning rate
        self.n_iterations = n_iterations  # Number of iterations
        self.reg_strength = regularization_strength  # Regularization strength
        self.weights = None  # Weights
        self.bias = None  # Bias

    # sigmioid activation function to transform linear outputs to probabilities
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid activation function

    # train the model using gradient descent with regularization
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # Get all unique classes
        self.weights = np.zeros((len(self.classes), n_features))  # Initialize weights to zero
        self.bias = np.zeros(len(self.classes))  # Initialize biases to zero

        for idx, cls in enumerate(self.classes):
            y_binary = np.where(y == cls, 1, 0)  # Convert target variable to binary classification
            for _ in range(self.n_iterations):
                linear_model = np.dot(X, self.weights[idx]) + self.bias[idx]  # Linear model
                y_predicted = self.sigmoid(linear_model)  # Calculate predicted values using sigmoid function

                # Compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_binary)) + self.reg_strength * self.weights[idx]
                db = (1 / n_samples) * np.sum(y_predicted - y_binary)

                # Update weights and biases
                self.weights[idx] -= self.learning_rate * dw
                self.bias[idx] -= self.learning_rate * db

    # predict the class labels
    def predict(self, X):
        linear_model = np.dot(X, self.weights.T) + self.bias  # Linear model for all classes
        y_predicted = self.sigmoid(linear_model)  # Calculate predicted values using sigmoid function
        return np.argmax(y_predicted, axis=1)  # Return the class with the highest probability as the predicted result
