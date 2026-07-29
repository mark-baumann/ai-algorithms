"""
Unit-Tests für den k-Nearest-Neighbors-Klassifikator.

Testet:
- Initialisierung und Parameter-Validierung
- Distanzberechnung (euklidisch, manhattan)
- fit (lazy learner)
- predict und predict_proba
- Kantenfälle (k=1, k > n_samples, eine Klasse)
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from knn_from_scratch import KNearestNeighbors


class TestKNNInit:
    """Tests für die __init__-Methode."""

    def test_default_parameters(self):
        """Standard-Parameter sollten gesetzt sein."""
        knn = KNearestNeighbors()
        assert knn.k == 3
        assert knn.metric == "euclidean"
        assert knn.X_train is None
        assert knn.y_train is None

    def test_custom_parameters(self):
        """Benutzerdefinierte Parameter sollten übernommen werden."""
        knn = KNearestNeighbors(k=5, metric="manhattan")
        assert knn.k == 5
        assert knn.metric == "manhattan"

    def test_invalid_k_zero(self):
        """k=0 sollte einen ValueError auslösen."""
        with pytest.raises(ValueError, match="k muss mindestens 1 sein"):
            KNearestNeighbors(k=0)

    def test_invalid_k_negative(self):
        """k=-1 sollte einen ValueError auslösen."""
        with pytest.raises(ValueError, match="k muss mindestens 1 sein"):
            KNearestNeighbors(k=-1)

    def test_invalid_metric(self):
        """Eine unbekannte Metrik sollte einen ValueError auslösen."""
        with pytest.raises(ValueError, match="metric muss 'euclidean' oder 'manhattan' sein"):
            KNearestNeighbors(metric="cosine")


class TestFit:
    """Tests für die fit-Methode."""

    def test_fit_returns_self(self):
        """fit() sollte self zurückgeben (Fluent Interface)."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 1, 0])
        knn = KNearestNeighbors(k=1)
        result = knn.fit(X, y)
        assert result is knn

    def test_fit_stores_data(self):
        """fit() sollte die Trainingsdaten speichern."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 1, 0])
        knn = KNearestNeighbors(k=1)
        knn.fit(X, y)
        assert knn.X_train is not None
        assert knn.y_train is not None
        assert len(knn.X_train) == 3
        assert len(knn.y_train) == 3

    def test_fit_converts_to_float64(self):
        """fit() sollte X in float64 konvertieren."""
        X = np.array([[0, 0], [1, 1]], dtype=np.int32)
        y = np.array([0, 1])
        knn = KNearestNeighbors()
        knn.fit(X, y)
        assert knn.X_train.dtype == np.float64


class TestDistances:
    """Tests für die _compute_distances-Methode."""

    def test_euclidean_same_point(self):
        """Distanz eines Punktes zu sich selbst sollte 0 sein."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 1, 0])
        knn = KNearestNeighbors(metric="euclidean")
        knn.fit(X, y)
        dist = knn._compute_distances(np.array([0, 0]))
        assert dist[0] == pytest.approx(0.0)

    def test_euclidean_known_distance(self):
        """Euklidische Distanz zwischen (0,0) und (3,4) sollte 5 sein."""
        X = np.array([[3, 4]])
        y = np.array([0])
        knn = KNearestNeighbors(metric="euclidean")
        knn.fit(X, y)
        dist = knn._compute_distances(np.array([0, 0]))
        assert dist[0] == pytest.approx(5.0)

    def test_manhattan_same_point(self):
        """Manhattan-Distanz eines Punktes zu sich selbst sollte 0 sein."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        knn = KNearestNeighbors(metric="manhattan")
        knn.fit(X, y)
        dist = knn._compute_distances(np.array([0, 0]))
        assert dist[0] == pytest.approx(0.0)

    def test_manhattan_known_distance(self):
        """Manhattan-Distanz zwischen (0,0) und (3,4) sollte 7 sein."""
        X = np.array([[3, 4]])
        y = np.array([0])
        knn = KNearestNeighbors(metric="manhattan")
        knn.fit(X, y)
        dist = knn._compute_distances(np.array([0, 0]))
        assert dist[0] == pytest.approx(7.0)

    def test_distance_shape(self):
        """_compute_distances sollte ein Array mit n_samples Einträgen zurückgeben."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 1, 0, 1])
        knn = KNearestNeighbors()
        knn.fit(X, y)
        dist = knn._compute_distances(np.array([1.5, 1.5]))
        assert len(dist) == 4


class TestPredict:
    """Tests für predict und predict_one."""

    def test_predict_shape(self):
        """predict sollte ein Array mit gleicher Länge wie X zurückgeben."""
        X, y = make_classification(
            n_samples=50, n_features=2, n_redundant=0, n_informative=2,
            random_state=42
        )
        knn = KNearestNeighbors(k=3)
        knn.fit(X, y)
        pred = knn.predict(X)
        assert len(pred) == len(X)
        assert isinstance(pred, np.ndarray)

    def test_predict_single_sample(self):
        """predict sollte auch für einen einzelnen Sample funktionieren."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        knn = KNearestNeighbors(k=1)
        knn.fit(X, y)
        pred = knn.predict(np.array([[0.1, 0.1]]))
        assert len(pred) == 1

    def test_predict_k1_exact_match(self):
        """Mit k=1 sollte der nächste Nachbar exakt vorhergesagt werden."""
        X = np.array([[0, 0], [5, 5]])
        y = np.array([0, 1])
        knn = KNearestNeighbors(k=1)
        knn.fit(X, y)
        # Punkt nahe bei (0,0) → Klasse 0
        assert knn.predict(np.array([[0.1, 0.1]]))[0] == 0
        # Punkt nahe bei (5,5) → Klasse 1
        assert knn.predict(np.array([[4.9, 4.9]]))[0] == 1

    def test_predict_k3_majority(self):
        """Mit k=3 sollte die Mehrheitsklasse gewählt werden."""
        X = np.array([[0, 0], [0.1, 0.1], [5, 5]])
        y = np.array([0, 0, 1])
        knn = KNearestNeighbors(k=3)
        knn.fit(X, y)
        # Die 3 nächsten sind alle Trainingspunkte → 2x Klasse 0, 1x Klasse 1 → Klasse 0
        pred = knn.predict(np.array([[0.05, 0.05]]))
        assert pred[0] == 0

    def test_predict_accuracy_above_chance(self):
        """Auf einem einfachen Datensatz sollte die Genauigkeit > 70% sein."""
        X, y = make_classification(
            n_samples=200, n_features=2, n_redundant=0, n_informative=2,
            random_state=42
        )
        knn = KNearestNeighbors(k=5)
        knn.fit(X[:150], y[:150])
        pred = knn.predict(X[150:])
        acc = np.mean(pred == y[150:])
        assert acc > 0.7, f"Genauigkeit {acc:.2%} zu niedrig"

    def test_predict_dtype(self):
        """predict sollte Integer-Werte zurückgeben."""
        X, y = make_classification(
            n_samples=50, n_features=2, n_redundant=0, n_informative=2,
            random_state=42
        )
        knn = KNearestNeighbors(k=3)
        knn.fit(X, y)
        pred = knn.predict(X)
        assert pred.dtype in (np.int32, np.int64)


class TestPredictProba:
    """Tests für predict_proba."""

    def test_predict_proba_shape(self):
        """predict_proba sollte (n_samples, n_classes) zurückgeben."""
        X, y = make_classification(
            n_samples=50, n_features=2, n_redundant=0, n_informative=2,
            random_state=42
        )
        knn = KNearestNeighbors(k=3)
        knn.fit(X, y)
        proba = knn.predict_proba(X)
        assert proba.shape == (50, 2)

    def test_predict_proba_sums_to_one(self):
        """Die Wahrscheinlichkeiten sollten sich zu 1 summieren."""
        X, y = make_classification(
            n_samples=20, n_features=2, n_redundant=0, n_informative=2,
            random_state=42
        )
        knn = KNearestNeighbors(k=3)
        knn.fit(X, y)
        proba = knn.predict_proba(X)
        for row in proba:
            assert row.sum() == pytest.approx(1.0)

    def test_predict_proba_k1(self):
        """Mit k=1 sollten die Wahrscheinlichkeiten 0 oder 1 sein."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 1, 0])
        knn = KNearestNeighbors(k=1)
        knn.fit(X, y)
        proba = knn.predict_proba(X)
        # Jede Zeile sollte genau eine 1 und eine 0 haben
        for row in proba:
            assert (row == 1.0).sum() == 1
            assert (row == 0.0).sum() == 1

    def test_predict_proba_values_in_range(self):
        """Alle Wahrscheinlichkeiten sollten zwischen 0 und 1 liegen."""
        X, y = make_classification(
            n_samples=30, n_features=2, n_redundant=0, n_informative=2,
            random_state=42
        )
        knn = KNearestNeighbors(k=5)
        knn.fit(X, y)
        proba = knn.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)


class TestEdgeCases:
    """Tests für Kantenfälle."""

    def test_single_class(self):
        """Bei nur einer Klasse sollte immer diese vorhergesagt werden."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([1, 1, 1])
        knn = KNearestNeighbors(k=2)
        knn.fit(X, y)
        pred = knn.predict(np.array([[0.5, 0.5], [10, 10]]))
        assert np.all(pred == 1)

    def test_k_larger_than_samples(self):
        """k > n_samples sollte trotzdem funktionieren (alle Samples werden genutzt)."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 0, 1])
        knn = KNearestNeighbors(k=10)
        knn.fit(X, y)
        pred = knn.predict(np.array([[0.5, 0.5]]))
        assert len(pred) == 1
        assert pred[0] in [0, 1]

    def test_float_input(self):
        """Float-Arrays sollten akzeptiert werden."""
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
        y = np.array([0, 1, 0])
        knn = KNearestNeighbors()
        knn.fit(X, y)
        pred = knn.predict(X)
        assert len(pred) == 3

    def test_many_features(self):
        """k-NN sollte auch mit vielen Features funktionieren."""
        X, y = make_classification(
            n_samples=100, n_features=20, n_redundant=5, n_informative=10,
            random_state=42
        )
        knn = KNearestNeighbors(k=5)
        knn.fit(X, y)
        pred = knn.predict(X)
        assert len(pred) == 100

    def test_three_classes(self):
        """k-NN sollte auch mit 3 Klassen funktionieren."""
        X, y = make_classification(
            n_samples=90, n_features=2, n_redundant=0, n_informative=2,
            n_classes=3, n_clusters_per_class=1, random_state=42
        )
        knn = KNearestNeighbors(k=5)
        knn.fit(X, y)
        pred = knn.predict(X)
        assert len(np.unique(pred)) >= 1
        proba = knn.predict_proba(X)
        assert proba.shape == (90, 3)
