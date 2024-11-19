import numpy as np
from sklearn.linear_model import LogisticRegression

class LocalModel:
    """
    Local Model for Worker Node.
    Handles local training, gradient computation, and model updates.
    """

    def __init__(self, data, target, target_column):
        """
        Initialize the local model with data and target.
        :param data: Local dataset (features and target).
        :param target: Target column for classification.
        :param target_column: Name of the target column.
        """
        self.data = data.drop(columns=[target_column])
        self.target = data[target_column]
        self.model = LogisticRegression()
        print("Local Model initialized.")

    def train(self):
        """
        Train the logistic regression model on local data.
        """
        self.model.fit(self.data, self.target)
        print("Local Model trained successfully.")

    def compute_gradient(self):
        """
        Compute gradients for the logistic regression model.
        :return: Gradient array.
        """
        X = self.data.values
        y = self.target.values
        predictions = self.model.predict_proba(X)[:, 1]
        gradient = np.dot(X.T, (predictions - y)) / len(y)
        print(f"Computed Gradient: {gradient}")
        return gradient

    def update_model(self, global_weights):
        """
        Update the local model with the global weights from Master Node.
        :param global_weights: Array of global weights.
        """
        num_features = len(self.data.columns)
        self.model.coef_ = np.array(global_weights[:num_features]).reshape(1, -1)
        self.model.intercept_ = np.array(global_weights[num_features:])
        print(f"Updated local model with global weights: {global_weights}")

    def get_weights(self):
        """
        Retrieve the current weights of the model.
        :return: Array of model weights including intercept.
        """
        coef = self.model.coef_.flatten()
        intercept = self.model.intercept_
        weights = np.concatenate([coef, intercept])
        print(f"Retrieved Local Model Weights: {weights}")
        return weights
