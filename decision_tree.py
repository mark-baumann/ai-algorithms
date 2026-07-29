"""
Entscheidungsbaum (Decision Tree) — Von Grund auf mit NumPy
============================================================

Ein Entscheidungsbaum teilt die Daten rekursiv anhand von Features auf.
Jeder Split maximiert den "Information Gain" — die Reduktion der Unreinheit.

Lernziele:
- Verstehen, wie Gini-Impurity und Entropy funktionieren
- Rekursive Aufteilung der Daten
- Warum Bäume zu Overfitting neigen (und wie man das verhindert)

Autor: Hermes Agent für Marks KI-Lernreise
Datum: 29.07.2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class DecisionTreeNode:
    """
    Ein Knoten im Entscheidungsbaum.

    Blattknoten (leaf=True):
        - value: die vorhergesagte Klasse

    Innere Knoten (leaf=False):
        - feature_index: welches Feature wird gesplittet
        - threshold: bei welchem Wert wird gesplittet
        - left, right: Kindknoten
    """

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # Nur für Blätter: die vorhergesagte Klasse
        self.is_leaf = False


class DecisionTree:
    """
    Entscheidungsbaum-Klassifikator — von Grund auf.

    Parameter
    ---------
    max_depth : int
        Maximale Tiefe des Baums (verhindert Overfitting)
    min_samples_split : int
        Mindestanzahl Samples für einen Split
    criterion : str
        'gini' oder 'entropy' für die Split-Qualität
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_classes = None

    def _impurity(self, y: np.ndarray) -> float:
        """
        Berechnet die Unreinheit eines Knotens.

        Gini-Impurity: 1 - Σ(p_i²)
        - Misst, wie "gemischt" die Klassen sind
        - 0 = alle Samples gehören zur gleichen Klasse (rein)
        - 0.5 = perfekt gemischt bei 2 Klassen

        Entropy: -Σ(p_i * log₂(p_i))
        - Misst den Informationsgehalt
        - 0 = rein, 1 = maximal gemischt (bei 2 Klassen)
        """
        # Klassenhäufigkeiten
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities**2)
        else:  # entropy
            # Vermeide log(0) — nur positive Wahrscheinlichkeiten
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Findet den besten Split für einen Knoten.

        Probiert jedes Feature und jeden möglichen Threshold aus,
        und wählt den Split mit dem höchsten Information Gain.

        Information Gain = Impurity(parent) - gewichtete Impurity(children)
        """
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None

        parent_impurity = self._impurity(y)
        best_gain = -1
        best_split = None

        for feature_idx in range(n_features):
            # Sortierte eindeutige Werte des Features
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # Split: links ≤ threshold, rechts > threshold
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # Beide Seiten müssen Samples haben
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Gewichtete Impurity der Kinder
                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_impurity = (
                    n_left / n_samples * left_impurity
                    + n_right / n_samples * right_impurity
                )

                # Information Gain
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
        """
        Baut den Baum rekursiv auf.

        Abbruchbedingungen:
        1. Alle Samples gehören zur gleichen Klasse → Blatt
        2. Maximale Tiefe erreicht → Blatt
        3. Kein gültiger Split mehr möglich → Blatt
        """
        node = DecisionTreeNode()

        # Abbruchbedingungen prüfen
        n_samples = X.shape[0]
        unique_classes = np.unique(y)

        if (
            len(unique_classes) == 1
            or depth >= self.max_depth
            or n_samples < self.min_samples_split
        ):
            node.is_leaf = True
            # Häufigste Klasse als Vorhersage
            counts = np.bincount(y.astype(int))
            node.value = int(np.argmax(counts))
            return node

        # Besten Split finden
        split = self._best_split(X, y)
        if split is None:
            node.is_leaf = True
            counts = np.bincount(y.astype(int))
            node.value = int(np.argmax(counts))
            return node

        # Split durchführen
        node.feature_index = split["feature_index"]
        node.threshold = split["threshold"]

        left_mask = X[:, split["feature_index"]] <= split["threshold"]
        right_mask = ~left_mask

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Trainiert den Entscheidungsbaum."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """Durchläuft den Baum für einen einzelnen Punkt."""
        if node.is_leaf:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sagt Klassen für mehrere Punkte vorher."""
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.root) for x in X])

    def print_tree(self, node: DecisionTreeNode = None, depth: int = 0):
        """Gibt den Baum als Text aus (für Debugging)."""
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}└─ Blatt: Klasse {node.value}")
        else:
            print(
                f"{indent}├─ Split: X[{node.feature_index}] ≤ {node.threshold:.2f}"
            )
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)


def plot_decision_boundary(
    model, X: np.ndarray, y: np.ndarray, title: str, ax: plt.Axes
):
    """Visualisiert die Entscheidungsgrenze."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#FF0000", "#0000FF"]),
        edgecolor="k", s=50
    )
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.set_xlabel("Merkmal 1")
    ax.set_ylabel("Merkmal 2")


def demonstrate_depth_effect():
    """
    Demonstriert, wie max_depth Overfitting/Underfitting beeinflusst.

    Tiefe 1 (Stumpf):
    - Nur ein Split → starkes Underfitting
    - Entscheidungsgrenze ist eine einzelne Linie

    Tiefe 10 (tief):
    - Viele Splits → Overfitting
    - "Zackige" Grenzen, die sich an einzelne Punkte anpassen
    """
    print("\n" + "=" * 60)
    print("📊 Demo: Einfluss von max_depth")
    print("=" * 60)

    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    depths = [1, 2, 3, 5, 10, 20]

    for ax, depth in zip(axes.flat, depths):
        model = DecisionTree(max_depth=depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        plot_decision_boundary(
            model, X_train, y_train,
            f"max_depth={depth} (Genauigkeit: {acc:.2%})", ax
        )

    plt.suptitle(
        "Entscheidungsbaum: Tiefe 1 → Underfitting | Tiefe 20 → Overfitting",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("decision_tree_depth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gespeichert: decision_tree_depth.png")


def demonstrate_criterion():
    """
    Vergleicht Gini vs. Entropy als Split-Kriterium.

    In der Praxis sind die Unterschiede meist gering.
    Gini ist etwas schneller (kein log), Entropy tendiert zu
    ausgewogeneren Splits.
    """
    print("\n" + "=" * 60)
    print("📊 Demo: Gini vs. Entropy")
    print("=" * 60)

    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, criterion in zip(axes, ["gini", "entropy"]):
        model = DecisionTree(max_depth=4, criterion=criterion)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        plot_decision_boundary(
            model, X_train, y_train,
            f"{criterion.capitalize()} (Genauigkeit: {acc:.2%})", ax
        )

    plt.suptitle(
        "Entscheidungsbaum: Gini vs. Entropy (fast identisch in der Praxis)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("decision_tree_criterion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gespeichert: decision_tree_criterion.png")


def main():
    print("=" * 60)
    print("🌳 Entscheidungsbaum — Von Grund auf mit NumPy")
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

    model = DecisionTree(max_depth=4, criterion="gini")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"  Trainingsdaten: {len(X_train)} Samples")
    print(f"  Testdaten:      {len(X_test)} Samples")
    print(f"  Genauigkeit:    {acc:.2%}")
    print(f"  max_depth:      {model.max_depth}")
    print(f"  Kriterium:      {model.criterion}")
    print("\n  Baumstruktur:")
    model.print_tree()

    # 2. Tiefen-Effekt
    demonstrate_depth_effect()

    # 3. Gini vs. Entropy
    demonstrate_criterion()

    # 4. Klassifikationsreport
    print("\n" + "=" * 60)
    print("📊 Klassifikationsreport (max_depth=4, gini)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Klasse 0", "Klasse 1"]))

    print("\n✅ Alle Demos abgeschlossen!")
    print("📁 Erstellte Dateien: decision_tree_depth.png, decision_tree_criterion.png")
    print()
    print("💡 Key Takeaways:")
    print("  1. Entscheidungsbäume teilen Daten rekursiv an Features")
    print("  2. Gini/Entropy misst die 'Reinheit' eines Knotens")
    print("  3. Information Gain = Reduktion der Unreinheit durch Split")
    print("  4. max_depth ist der wichtigste Hyperparameter gegen Overfitting")
    print("  5. Bäume sind interpretierbar — man kann die Regeln ablesen!")


if __name__ == "__main__":
    main()
