import numpy as np


class LinearSVM:
    """
    lr : Learning rate
    C : Regularization strength
    """

    def __init__(self, lr=0.001, C=1.0, epochs=1000):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_his = []

    def fit(self, X, y):
        """
        X : ndarray (n_samples, n_features)
        y : ndarray (n_samples,)
            Labels must be {-1, +1}
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_his = []  # Reset history for new training

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):

                margin = y[idx] * (np.dot(x_i, self.w) + self.b)

                if margin >= 1:
                    # Only regularization term contributes
                    dw = self.w
                    db = 0
                else:
                    # Hinge loss active
                    dw = self.w - self.C * y[idx] * x_i
                    db = -self.C * y[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

            # Compute and store loss once per epoch
            loss = self.compute_loss(X, y)
            self.loss_his.append(loss)
            
            # Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def compute_loss(self, X, y):
        n_samples = X.shape[0]
        margins = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.maximum(0, 1 - margins)
        total_loss = 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_loss)
        return total_loss / n_samples

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))
