"""
Plot-Utilities für KI-Algorithmen
=================================
Visualisierungsfunktionen für Decision Boundaries und andere Plots.

Verwendung:
    from plot_utils import plot_decision_boundary
    plot_decision_boundary(model, X, y, "Titel", ax)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, title, ax=None):
    """
    Zeichnet die Decision Boundary eines Klassifikators.

    Args:
        model: Klassifikator mit .predict(X) Methode
        X: Feature-Matrix (n_samples, 2)
        y: Labels (n_samples,)
        title: Plot-Titel
        ax: Matplotlib-Achse (optional)
    """
    if ax is None:
        ax = plt.gca()

    # Grid erstellen
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Vorhersagen für das Grid
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Decision Boundary zeichnen
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="RdYlBu")

    ax.set_xlabel("Merkmal 1")
    ax.set_ylabel("Merkmal 2")
    ax.set_title(title)
