"""
Gemeinsame Visualisierungs-Hilfsfunktionen für ML-Algorithmen.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_boundary(
    model, X: np.ndarray, y: np.ndarray, title: str, ax: plt.Axes
):
    """
    Visualisiert die Entscheidungsgrenze eines Klassifikators.

    So funktioniert's:
    1. Ein feines Gitter über den gesamten Feature-Raum legen
    2. Für jeden Gitterpunkt die Klasse vorhersagen
    3. Die Flächen entsprechend einfärben
    4. Die echten Datenpunkte darüber plotten
    """
    h = 0.02  # Schrittweite des Gitters

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Alle Gitterpunkte klassifizieren
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Entscheidungsregionen einfärben
    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

    # Trainingsdaten plotten
    ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#FF0000", "#0000FF"]),
        edgecolor="k", s=50
    )
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.set_xlabel("Merkmal 1")
    ax.set_ylabel("Merkmal 2")
