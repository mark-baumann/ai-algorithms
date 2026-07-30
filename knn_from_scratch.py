"""
k-Nearest-Neighbors von Grund auf
================================
Implementierung des k-NN-Algorithmus ohne externe ML-Bibliotheken.
Unterstützt euklidische und Manhattan-Distanz.

Verwendung:
    from knn_from_scratch import KNearestNeighbors
    knn = KNearestNeighbors(k=3, metric="euclidean")
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
"""

import numpy as np
from collections import Counter


class KNearestNeighbors:
    """
    k-Nearest-Neighbors-Klassifikator.

    Args:
        k: Anzahl der Nachbarn (≥ 1)
        metric: "euclidean" oder "manhattan"
    """

    def __init__(self, k: int = 3, metric: str = "euclidean"):
        if k < 1:
            raise ValueError("k muss mindestens 1 sein")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric muss 'euclidean' oder 'manhattan' sein")
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Speichert die Trainingsdaten (Lazy Learner)."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        return self

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Berechnet Distanzen von x zu allen Trainingspunkten."""
        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        else:  # manhattan
            return np.sum(np.abs(self.X_train - x), axis=1)

    def _predict_one(self, x: np.ndarray) -> int:
        """Sagt die Klasse für einen einzelnen Punkt vorher."""
        distances = self._compute_distances(x)
        k_nearest_idx = np.argpartition(distances, min(self.k, len(distances) - 1))[:self.k]
        k_nearest_labels = self.y_train[k_nearest_idx]
        # Mehrheitsentscheidung
        counter = Counter(k_nearest_labels)
        return counter.most_common(1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sagt Klassen für mehrere Punkte vorher."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_one(x) for x in X], dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Gibt Klassenwahrscheinlichkeiten zurück."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        proba = np.zeros((len(X), n_classes))

        for i, x in enumerate(X):
            distances = self._compute_distances(x)
            k_nearest_idx = np.argpartition(distances, min(self.k, len(distances) - 1))[:self.k]
            k_nearest_labels = self.y_train[k_nearest_idx]
            for j, cls in enumerate(classes):
                proba[i, j] = np.sum(k_nearest_labels == cls) / self.k

        return proba
