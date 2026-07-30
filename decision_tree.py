"""
Entscheidungsbaum von Grund auf
===============================
Implementierung eines Entscheidungsbaum-Klassifikators ohne externe ML-Bibliotheken.
Unterstützt Gini-Impurity und Entropy als Split-Kriterien.

Verwendung:
    from decision_tree import DecisionTree, DecisionTreeNode
    tree = DecisionTree(max_depth=5, criterion="gini")
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
"""

import numpy as np
from collections import Counter


class DecisionTreeNode:
    """Ein Knoten im Entscheidungsbaum."""

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = False


class DecisionTree:
    """
    Entscheidungsbaum-Klassifikator.

    Args:
        max_depth: Maximale Tiefe des Baums
        min_samples_split: Minimale Samples für einen Split
        criterion: "gini" oder "entropy"
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2,
                 criterion: str = "gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_classes = None

    def _impurity(self, y: np.ndarray) -> float:
        """Berechnet die Impurity (Gini oder Entropy)."""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)

        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        else:  # entropy
            # Vermeide log(0)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> dict | None:
        """Findet den besten Split für die Daten."""
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None

        parent_impurity = self._impurity(y)
        best_gain = -1
        best_split = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_impurity = (n_left / n_samples) * left_impurity + \
                                   (n_right / n_samples) * right_impurity
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature_index": feature_idx,
                        "threshold": threshold,
                        "gain": gain,
                    }

        return best_split

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        """Rekursiver Baum-Aufbau."""
        node = DecisionTreeNode()

        # Abbruchbedingungen: max_depth erreicht oder nur eine Klasse
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node

        split = self._best_split(X, y)
        if split is None:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node

        node.feature_index = split["feature_index"]
        node.threshold = split["threshold"]

        left_mask = X[:, node.feature_index] <= node.threshold
        right_mask = ~left_mask

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trainiert den Entscheidungsbaum."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """Sagt die Klasse für einen einzelnen Punkt vorher."""
        if node.is_leaf:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sagt Klassen für mehrere Punkte vorher."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_one(x, self.root) for x in X], dtype=np.int64)

    def print_tree(self, node: DecisionTreeNode = None, depth: int = 0):
        """Gibt den Baum als Text aus."""
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}Blatt: Klasse {node.value}")
        else:
            print(f"{indent}Split: Feature {node.feature_index} <= {node.threshold:.3f}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)
