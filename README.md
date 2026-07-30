# KI-Algorithmen — eine Notebook-Reihe von Grund auf

Eine in sich geschlossene Reihe von neun Jupyter-Notebooks, die zentrale Algorithmen des
maschinellen Lernens aus ihren mathematischen Grundlagen herleitet, mit `NumPy`
implementiert, empirisch untersucht und gegen `scikit-learn` validiert.

Jedes Notebook folgt derselben Struktur: Zusammenfassung und Lernziele, theoretische
Herleitung, Implementierung, empirische Analyse mit Diskussion der Ergebnisse, Grenzen
des Verfahrens, Übungsaufgaben und Literaturangaben. Alle Notebooks sind vollständig
ausführbar (`Kernel → Restart & Run All`) und verwenden ausschließlich simulierte oder
über `scikit-learn` mitgelieferte Datensätze — es sind keine externen Downloads
erforderlich.

## Inhalt

| # | Notebook | Thema |
|---|----------|-------|
| 00 | [`00_grundlagen.ipynb`](notebooks/00_grundlagen.ipynb) | Lineare Algebra, Wahrscheinlichkeit, Optimierung, Bias-Varianz-Zerlegung |
| 01 | [`01_lineare_regression.ipynb`](notebooks/01_lineare_regression.ipynb) | Kleinste Quadrate, Normalengleichungen, Gradientenabstieg, Inferenz |
| 02 | [`02_logistische_regression.ipynb`](notebooks/02_logistische_regression.ipynb) | Maximum-Likelihood-Klassifikation, ROC-Analyse, Kalibrierung |
| 03 | [`03_k_nearest_neighbors.ipynb`](notebooks/03_k_nearest_neighbors.ipynb) | Nichtparametrische Klassifikation, Fluch der Dimensionalität |
| 04 | [`04_entscheidungsbaum.ipynb`](notebooks/04_entscheidungsbaum.ipynb) | Rekursive Partitionierung, Gini/Entropie, Overfitting |
| 05 | [`05_naive_bayes.ipynb`](notebooks/05_naive_bayes.ipynb) | Bayes-Theorem, Gaussian & Multinomial Naive Bayes, Textklassifikation |
| 06 | [`06_k_means.ipynb`](notebooks/06_k_means.ipynb) | Unüberwachtes Clustering, Lloyds Algorithmus, k-means++ |
| 07 | [`07_pca.ipynb`](notebooks/07_pca.ipynb) | Hauptkomponentenanalyse, Dimensionsreduktion, SVD |
| 08 | [`08_neuronales_netz.ipynb`](notebooks/08_neuronales_netz.ipynb) | Mehrschichtiges Perzeptron, Backpropagation, Gradient Checking |

Notebook 00 legt das mathematische Vokabular (lineare Algebra, Wahrscheinlichkeit,
Optimierung, Bias-Varianz-Tradeoff), auf das die Notebooks 01–08 explizit verweisen.
Die überwachten Modelle (01–05, 08) werden konsequent auf denselben Beispieldatensätzen
verglichen; 06–07 behandeln unüberwachtes Lernen.

## Verwendete Werkzeuge

Alle Algorithmen sind vollständig mit `NumPy` implementiert. `scikit-learn` dient
ausschließlich zur Erzeugung von Datensätzen, zur Validierung der Eigenimplementierungen
und als Vergleichsmaßstab — nicht als Ersatz für die eigene Herleitung. `pandas` und
`matplotlib` werden für Tabellen bzw. Visualisierungen verwendet, `scipy` für
Verteilungsfunktionen und numerische Optimierung in Notebook 00.

## Installation

```bash
git clone https://github.com/mark-baumann/ki-algorithmen.git
cd ki-algorithmen

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Nutzung

```bash
jupyter notebook notebooks/
```

Die Notebooks sind nummeriert und bauen inhaltlich aufeinander auf (00 → 08); sie können
aber auch einzeln gelesen werden, da jedes Notebook die benötigten Vorkenntnisse mit
Verweis auf die entsprechende Stelle in Notebook 00 kurz einführt.

## Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)
