"""
Unit-Tests für den Entscheidungsbaum-Klassifikator.

Testet:
- Initialisierung und Parameter-Validierung
- Gini-Impurity und Entropy-Berechnung
- Baum-Aufbau (fit)
- Vorhersage (predict)
- Kantenfälle (eine Klasse, leere Daten, min_samples_split)
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from app.decision_tree import DecisionTree, DecisionTreeNode


class TestDecisionTreeNode:
    """Tests für die DecisionTreeNode-Datenklasse."""

    def test_initial_state(self):
        """Ein neuer Knoten sollte alle Attribute auf None/False haben."""
        node = DecisionTreeNode()
        assert node.feature_index is None
        assert node.threshold is None
        assert node.left is None
        assert node.right is None
        assert node.value is None
        assert node.is_leaf is False


class TestDecisionTreeInit:
    """Tests für die __init__-Methode."""

    def test_default_parameters(self):
        """Standard-Parameter sollten gesetzt sein."""
        tree = DecisionTree()
        assert tree.max_depth == 5
        assert tree.min_samples_split == 2
        assert tree.criterion == "gini"
        assert tree.root is None
        assert tree.n_classes is None

    def test_custom_parameters(self):
        """Benutzerdefinierte Parameter sollten übernommen werden."""
        tree = DecisionTree(max_depth=3, min_samples_split=5, criterion="entropy")
        assert tree.max_depth == 3
        assert tree.min_samples_split == 5
        assert tree.criterion == "entropy"


class TestImpurity:
    """Tests für die _impurity-Methode."""

    def test_gini_pure(self):
        """Gini einer reinen Klasse sollte 0 sein."""
        tree = DecisionTree(criterion="gini")
        y = np.array([0, 0, 0, 0])
        assert tree._impurity(y) == pytest.approx(0.0)

    def test_gini_balanced(self):
        """Gini bei zwei gleich verteilten Klassen sollte 0.5 sein."""
        tree = DecisionTree(criterion="gini")
        y = np.array([0, 0, 1, 1])
        assert tree._impurity(y) == pytest.approx(0.5)

    def test_gini_three_classes(self):
        """Gini bei drei gleich verteilten Klassen."""
        tree = DecisionTree(criterion="gini")
        y = np.array([0, 1, 2])
        expected = 1.0 - (1/3)**2 * 3  # = 1 - 1/3 = 2/3
        assert tree._impurity(y) == pytest.approx(expected)

    def test_entropy_pure(self):
        """Entropy einer reinen Klasse sollte 0 sein."""
        tree = DecisionTree(criterion="entropy")
        y = np.array([0, 0, 0, 0])
        assert tree._impurity(y) == pytest.approx(0.0, abs=1e-9)

    def test_entropy_balanced(self):
        """Entropy bei zwei gleich verteilten Klassen sollte 1.0 sein."""
        tree = DecisionTree(criterion="entropy")
        y = np.array([0, 0, 1, 1])
        assert tree._impurity(y) == pytest.approx(1.0)


class TestFit:
    """Tests für die fit-Methode."""

    def test_fit_returns_self(self):
        """fit() sollte self zurückgeben (Fluent Interface)."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTree(max_depth=2)
        result = tree.fit(X, y)
        assert result is tree

    def test_fit_sets_n_classes(self):
        """Nach fit sollte n_classes gesetzt sein."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTree()
        tree.fit(X, y)
        assert tree.n_classes == 2

    def test_fit_creates_root(self):
        """Nach fit sollte root ein DecisionTreeNode sein."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTree()
        tree.fit(X, y)
        assert isinstance(tree.root, DecisionTreeNode)

    def test_fit_single_class(self):
        """Bei nur einer Klasse sollte der Baum ein einzelnes Blatt sein."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 0, 0])
        tree = DecisionTree()
        tree.fit(X, y)
        assert tree.root.is_leaf
        assert tree.root.value == 0

    def test_fit_max_depth_zero(self):
        """Mit max_depth=0 sollte der Baum sofort ein Blatt sein."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTree(max_depth=0)
        tree.fit(X, y)
        assert tree.root.is_leaf

    def test_fit_min_samples_split_high(self):
        """Mit hohem min_samples_split sollte der Baum ein Blatt sein."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTree(min_samples_split=10)
        tree.fit(X, y)
        assert tree.root.is_leaf


class TestPredict:
    """Tests für die predict-Methode."""

    def test_predict_shape(self):
        """predict sollte ein Array mit gleicher Länge wie X zurückgeben."""
        X, y = make_classification(
            n_samples=50, n_features=2, n_redundant=0, n_informative=2,
            random_state=42
        )
        tree = DecisionTree(max_depth=3)
        tree.fit(X, y)
        pred = tree.predict(X)
        assert len(pred) == len(X)
        assert isinstance(pred, np.ndarray)

    def test_predict_single_sample(self):
        """predict sollte auch für einen einzelnen Sample funktionieren."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)
        pred = tree.predict(np.array([[1.5, 1.5]]))
        assert len(pred) == 1
        assert pred[0] in [0, 1]

    def test_predict_all_same_class(self):
        """Wenn alle Trainingsdaten die gleiche Klasse haben, sollte nur diese vorhergesagt werden."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([1, 1, 1])
        tree = DecisionTree()
        tree.fit(X, y)
        pred = tree.predict(np.array([[0.5, 0.5], [10, 10]]))
        assert np.all(pred == 1)

    def test_predict_accuracy_above_chance(self):
        """Auf einem einfachen Datensatz sollte die Genauigkeit > 70% sein."""
        X, y = make_classification(
            n_samples=200, n_features=2, n_redundant=0, n_informative=2,
            random_state=42
        )
        tree = DecisionTree(max_depth=5)
        tree.fit(X[:150], y[:150])
        pred = tree.predict(X[150:])
        acc = np.mean(pred == y[150:])
        assert acc > 0.7, f"Genauigkeit {acc:.2%} zu niedrig"

    def test_predict_dtype(self):
        """predict sollte Integer-Werte zurückgeben."""
        X, y = make_classification(
            n_samples=50, n_features=2, n_redundant=0, n_informative=2,
            random_state=42
        )
        tree = DecisionTree(max_depth=3)
        tree.fit(X, y)
        pred = tree.predict(X)
        assert pred.dtype in (np.int32, np.int64)


class TestBestSplit:
    """Tests für die _best_split-Methode."""

    def test_best_split_too_few_samples(self):
        """Bei zu wenigen Samples sollte None zurückgegeben werden."""
        tree = DecisionTree(min_samples_split=10)
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        result = tree._best_split(X, y)
        assert result is None

    def test_best_split_returns_dict(self):
        """Ein gültiger Split sollte ein dict mit feature_index, threshold, gain sein."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTree()
        result = tree._best_split(X, y)
        assert result is not None
        assert "feature_index" in result
        assert "threshold" in result
        assert "gain" in result
        assert result["gain"] >= 0


class TestPrintTree:
    """Tests für die print_tree-Methode (Ausgabe-Prüfung)."""

    def test_print_tree_does_not_crash(self, capsys):
        """print_tree sollte ohne Fehler durchlaufen."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)
        tree.print_tree()
        captured = capsys.readouterr()
        assert "Split" in captured.out or "Blatt" in captured.out


class TestEdgeCases:
    """Tests für Kantenfälle."""

    def test_float_input(self):
        """Float-Arrays sollten akzeptiert werden."""
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
        y = np.array([0, 0, 1, 1])
        tree = DecisionTree()
        tree.fit(X, y)
        pred = tree.predict(X)
        assert len(pred) == 4

    def test_many_features(self):
        """Der Baum sollte auch mit vielen Features funktionieren."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_redundant=3, n_informative=5,
            random_state=42
        )
        tree = DecisionTree(max_depth=3)
        tree.fit(X, y)
        pred = tree.predict(X)
        assert len(pred) == 100

    def test_three_classes(self):
        """Der Baum sollte auch mit 3 Klassen funktionieren."""
        X, y = make_classification(
            n_samples=150, n_features=2, n_redundant=0, n_informative=2,
            n_classes=3, n_clusters_per_class=1, random_state=42
        )
        tree = DecisionTree(max_depth=5)
        tree.fit(X, y)
        pred = tree.predict(X)
        assert len(np.unique(pred)) >= 1
        assert tree.n_classes == 3
