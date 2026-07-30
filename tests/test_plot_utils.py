"""
Unit-Tests für plot_utils.py.

Testet:
- plot_decision_boundary läuft ohne Fehler
- Achsenbeschriftungen werden gesetzt
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Kein GUI-Backend nötig
import matplotlib.pyplot as plt

from plot_utils import plot_decision_boundary


class MockModel:
    """Ein Mock-Klassifikator für Tests."""

    def predict(self, X):
        # Einfache Regel: Klasse 0 wenn x0 + x1 < 0, sonst Klasse 1
        return (X[:, 0] + X[:, 1] >= 0).astype(int)


class TestPlotDecisionBoundary:
    """Tests für plot_decision_boundary."""

    def test_creates_plot(self):
        """Die Funktion sollte ohne Fehler durchlaufen."""
        X = np.array([[0, 0], [1, 1], [-1, -1], [2, -1]])
        y = np.array([0, 1, 0, 1])
        model = MockModel()

        fig, ax = plt.subplots()
        plot_decision_boundary(model, X, y, "Test Plot", ax)
        plt.close(fig)

    def test_sets_title(self):
        """Der übergebene Titel sollte gesetzt werden."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        model = MockModel()

        fig, ax = plt.subplots()
        plot_decision_boundary(model, X, y, "Mein Titel", ax)
        assert ax.get_title() == "Mein Titel"
        plt.close(fig)

    def test_sets_labels(self):
        """Achsenbeschriftungen sollten gesetzt sein."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        model = MockModel()

        fig, ax = plt.subplots()
        plot_decision_boundary(model, X, y, "Test", ax)
        assert ax.get_xlabel() == "Merkmal 1"
        assert ax.get_ylabel() == "Merkmal 2"
        plt.close(fig)

    def test_handles_single_class(self):
        """Sollte auch mit nur einer Klasse funktionieren."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 0, 0])
        model = MockModel()

        fig, ax = plt.subplots()
        plot_decision_boundary(model, X, y, "Eine Klasse", ax)
        plt.close(fig)

    def test_handles_many_points(self):
        """Sollte auch mit vielen Punkten funktionieren."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = MockModel()

        fig, ax = plt.subplots()
        plot_decision_boundary(model, X, y, "Viele Punkte", ax)
        plt.close(fig)
