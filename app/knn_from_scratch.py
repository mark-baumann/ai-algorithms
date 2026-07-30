"""
k-Nearest Neighbors (k-NN) — Von Grund auf mit NumPy
=====================================================

Der k-NN-Algorithmus ist einer der einfachsten ML-Algorithmen:
- Für einen neuen Datenpunkt finden wir die k nächsten Nachbarn im Trainingsset
- Die Klasse wird durch Mehrheitsentscheid der Nachbarn bestimmt

Lernziele:
- Verstehen, wie Distanzmetriken funktionieren (Euklidisch, Manhattan)
- Verstehen, wie k die Entscheidungsgrenze beeinflusst
- Implementierung ohne Scikit-Learn — nur NumPy!

Autor: Hermes Agent für Marks KI-Lernreise
Datum: 29.07.2026
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from app.plot_utils import plot_decision_boundary


class KNearestNeighbors:
    """
    k-NN Klassifikator — von Grund auf implementiert.

    Parameter
    ---------
    k : int
        Anzahl der Nachbarn (Standard: 3)
    metric : str
        Distanzmetrik: 'euclidean' oder 'manhattan' (Standard: 'euclidean')
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        """
        k-NN ist ein "lazy learner" — fit speichert nur die Trainingsdaten.
        Kein echtes Training nötig!
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        return self

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Berechnet die Distanz von einem Punkt x zu ALLEN Trainingspunkten.

        Statt einer langsamen for-Schleife nutzen wir Vektorisierung:
        - Euklidisch: sqrt(sum((x - X_train)²))
        - Manhattan: sum(|x - X_train|)
        """
        if self.metric == "euclidean":
            # Vektorisierte euklidische Distanz
            # ||x - X_train||₂ = sqrt(sum((x - X_train)², axis=1))
            diff = self.X_train - x  # Broadcasting: (n_samples, n_features)
            distances = np.sqrt(np.sum(diff**2, axis=1))
        else:  # manhattan
            diff = self.X_train - x
            distances = np.sum(np.abs(diff), axis=1)
        return distances

    def predict_one(self, x: np.ndarray) -> int:
        """
        Sagt die Klasse für EINEN einzelnen Punkt vorher.

        Ablauf:
        1. Distanzen zu allen Trainingspunkten berechnen
        2. Die k nächsten Nachbarn finden (argsort)
        3. Mehrheitsentscheid der k Nachbarn
        """
        distances = self._compute_distances(x)

        # Indizes der k nächsten Nachbarn
        # np.argsort gibt die Indizes zurück, die das Array sortieren würden
        nearest_indices = np.argsort(distances)[: self.k]

        # Klassen der k nächsten Nachbarn
        nearest_labels = self.y_train[nearest_indices]

        # Mehrheitsentscheid: häufigste Klasse
        # np.bincount zählt Häufigkeiten, argmax gibt den Index der häufigsten
        unique, counts = np.unique(nearest_labels, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sagt Klassen für mehrere Punkte vorher."""
        X = np.asarray(X, dtype=np.float64)
        return np.array([self.predict_one(x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Gibt Wahrscheinlichkeiten zurück: Anteil der k Nachbarn pro Klasse.

        Das ist nützlich für:
        - ROC-Kurven
        - Unsicherheitsabschätzung
        - Soft-Voting in Ensembles
        """
        X = np.asarray(X, dtype=np.float64)
        classes = np.unique(self.y_train)
        probas = np.zeros((len(X), len(classes)))

        for i, x in enumerate(X):
            distances = self._compute_distances(x)
            nearest_indices = np.argsort(distances)[: self.k]
            nearest_labels = self.y_train[nearest_indices]
            for j, cls in enumerate(classes):
                probas[i, j] = np.sum(nearest_labels == cls) / self.k

        return probas


def demonstrate_k_effect():
    """
    Demonstriert, wie k die Entscheidungsgrenze beeinflusst.

    Kleines k (z.B. k=1):
    - Sehr flexible, "zackige" Grenzen
    - Overfitting-Gefahr: jeder Ausreißer wird mitgenommen

    Großes k (z.B. k=15):
    - Glattere, generalisiertere Grenzen
    - Underfitting-Gefahr: feine Strukturen gehen verloren
    """
    print("\n" + "=" * 60)
    print("📊 Demo: Einfluss von k auf die Entscheidungsgrenze")
    print("=" * 60)

    # Datensatz: zwei verschlungene Halbmonde
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    _fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    k_values = [1, 3, 5, 7, 11, 21]

    for ax, k in zip(axes.flat, k_values):
        model = KNearestNeighbors(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        plot_decision_boundary(
            model, X_train, y_train,
            f"k={k} (Test-Genauigkeit: {acc:.2%})", ax
        )

    plt.suptitle(
        "k-NN: Kleines k → Overfitting | Großes k → Underfitting",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("knn_k_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gespeichert: knn_k_effect.png")


def demonstrate_metrics():
    """
    Demonstriert den Unterschied zwischen euklidischer und Manhattan-Distanz.

    Euklidisch (L2): "Luftlinie" — sqrt(Σ(x_i - y_i)²)
    Manhattan (L1): "Taxi-Distanz" — Σ|x_i - y_i|

    Bei hochdimensionalen Daten ist Manhattan oft robuster,
    weil die euklidische Distanz mit der Dimension "explodiert"
    (Curse of Dimensionality).
    """
    print("\n" + "=" * 60)
    print("📊 Demo: Euklidische vs. Manhattan-Distanz")
    print("=" * 60)

    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["euclidean", "manhattan"]):
        model = KNearestNeighbors(k=5, metric=metric)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        plot_decision_boundary(
            model, X_train, y_train,
            f"{metric.capitalize()}-Distanz (Genauigkeit: {acc:.2%})", ax
        )

    plt.suptitle(
        "k-NN: Euklidisch (L2) vs. Manhattan (L1)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("knn_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gespeichert: knn_metrics.png")


def demonstrate_datasets():
    """
    Testet k-NN auf verschiedenen Dataset-Typen.

    - Linear trennbar: k-NN funktioniert gut
    - Kreisförmig: k-NN braucht genug Samples
    - Moons: k-NN kann nicht-lineare Grenzen lernen
    """
    print("\n" + "=" * 60)
    print("📊 Demo: k-NN auf verschiedenen Datensätzen")
    print("=" * 60)

    datasets = [
        ("Linear trennbar", make_classification(
            n_samples=200, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, random_state=42
        )),
        ("Kreise", make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)),
        ("Halbmonde", make_moons(n_samples=200, noise=0.15, random_state=42)),
    ]

    _fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, (X, y)) in zip(axes, datasets):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = KNearestNeighbors(k=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        plot_decision_boundary(
            model, X_train, y_train,
            f"{name}\nGenauigkeit: {acc:.2%}", ax
        )

    plt.suptitle(
        "k-NN: Funktioniert auf allen Dataset-Typen (nicht-parametrisch)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("knn_datasets.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gespeichert: knn_datasets.png")


def main():
    print("=" * 60)
    print("🧠 k-Nearest Neighbors — Von Grund auf mit NumPy")
    print("=" * 60)

    # 1. Einfacher Test
    print("\n📌 1. Einfacher Funktionstest")
    X, y = make_classification(
        n_samples=100, n_features=2, n_redundant=0, n_informative=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = KNearestNeighbors(k=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"  Trainingsdaten: {len(X_train)} Samples")
    print(f"  Testdaten:      {len(X_test)} Samples")
    print(f"  Genauigkeit:    {acc:.2%}")
    print(f"  k:              {model.k}")
    print(f"  Metrik:         {model.metric}")

    # 2. k-Effekt demonstrieren
    demonstrate_k_effect()

    # 3. Metriken vergleichen
    demonstrate_metrics()

    # 4. Verschiedene Datensätze
    demonstrate_datasets()

    # 5. Klassifikationsreport
    print("\n" + "=" * 60)
    print("📊 Klassifikationsreport (k=5, euklidisch)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Klasse 0", "Klasse 1"]))

    print("\n✅ Alle Demos abgeschlossen!")
    print("📁 Erstellte Dateien: knn_k_effect.png, knn_metrics.png, knn_datasets.png")
    print()
    print("💡 Key Takeaways:")
    print("  1. k-NN ist ein 'lazy learner' — kein Training, nur Speichern")
    print("  2. Kleines k → Overfitting (zackige Grenzen)")
    print("  3. Großes k → Underfitting (glatte, aber ungenaue Grenzen)")
    print("  4. Euklidisch = Luftlinie, Manhattan = Taxi-Distanz")
    print("  5. k-NN ist nicht-parametrisch: keine Annahmen über die Daten")


if __name__ == "__main__":
    main()
