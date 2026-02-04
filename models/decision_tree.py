import numpy as np


class DecisionTree:
    """
    max_depth : Maximum depth of tree
    min_samples_split : Minimum samples required to split
    """

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.feature_usage = np.zeros(self.n_features)
        self.tree = self._grow_tree(X, y)
        # Normalize feature importances
        if np.sum(self.feature_usage) > 0:
            self.feature_importances_ = self.feature_usage / np.sum(self.feature_usage)
        else:
            self.feature_importances_ = self.feature_usage

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return {"type": "leaf", "value": leaf_value}

        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            leaf_value = self._most_common_label(y)
            return {"type": "leaf", "value": leaf_value}

        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        # Track feature usage for importance calculation
        self.feature_usage[feature_idx] += n_samples

        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "type": "node",
            "feature": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }
    
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = 1.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gini = self._weighted_gini(y[left_mask], y[right_mask])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0

        counts = np.bincount(y)
        probabilities = counts / m
        return 1 - np.sum(probabilities ** 2)

    def _weighted_gini(self, y_left, y_right):
        m = len(y_left) + len(y_right)
        return (
            (len(y_left) / m) * self._gini(y_left)
            + (len(y_right) / m) * self._gini(y_right)
        )

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def _predict_sample(self, x, node):
        if node["type"] == "leaf":
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])
