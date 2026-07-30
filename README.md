# 🤖 KI-Algorithmen — Von Grund auf implementiert

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Aktiv-brightgreen.svg)]()

Fundamentale **Machine-Learning-Algorithmen** — von Grund auf mit NumPy implementiert und interaktiv visualisiert. Linear Regression, Logistic Regression mit Decision Boundaries, k-Nearest Neighbors und Decision Trees — verstehe die Mathematik hinter den Algorithmen, nicht nur die API.

## ✨ Features

- **📈 Linear Regression** — Daten generieren, Modell fitten, wahre vs. gelernte Linie vergleichen
- **📊 Logistic Regression** — Decision Boundary auf verschiedenen Datasets (linear trennbar, Moons, Kreise)
- **🔍 k-Nearest Neighbors** — Von Grund auf implementiert, mit verschiedenen k-Werten und Distanzmetriken
- **🌳 Decision Tree** — Selbst gebaut mit Gini-Impurity und Entropy, Baumstruktur anzeigen
- **🎨 Decision-Boundary-Visualisierung** — Farbige Regionen zeigen, wie das Modell den Raum aufteilt
- **📊 W&B-Integration** — Experiment-Tracking mit Weights & Biases
- **✅ Vollständig getestet** — Unit-Tests für k-NN, Decision Tree und Utilities

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/mark-baumann/ki-algorithmen.git
cd ki-algorithmen

# Virtuelle Umgebung erstellen
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Abhängigkeiten installieren
uv pip install -e ".[dev]"
```

## 🎯 Nutzung

```bash
# Streamlit-App starten
streamlit run app.py
```

Die App öffnet sich im Browser unter `http://localhost:8501`. Wähle einen der vier Tabs und experimentiere mit Datasets und Hyperparametern.

## 🧪 Tests ausführen

```bash
pytest tests/ -v
```

## 🛠️ Tech-Stack

| Technologie | Einsatz |
|-------------|---------|
| **NumPy** | Kern-Implementierung von k-NN und Decision Tree |
| **scikit-learn** | Linear/Logistic Regression, Datasets (make_blobs, make_moons) |
| **Matplotlib** | Decision Boundaries und Datenvisualisierung |
| **Streamlit** | Interaktive Web-App |
| **Weights & Biases** | Experiment-Tracking |
| **Pytest** | Test-Framework |

## 📁 Projektstruktur

```
ki-algorithmen/
├── app.py                  # Streamlit-Hauptapp (4 Tabs)
├── pyproject.toml          # Projekt-Konfiguration
├── knn_from_scratch.py     # k-NN Eigenimplementierung
├── decision_tree.py        # Decision Tree Eigenimplementierung
├── plot_utils.py           # Decision-Boundary-Visualisierung
├── wandb_utils.py          # W&B-Integration
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_knn.py
    ├── test_decision_tree.py
    ├── test_plot_utils.py
    └── test_wandb_utils.py
```

## 📖 Algorithmen im Detail

| Algorithmus | Typ | Komplexität | Selbst gebaut? |
|-------------|-----|-------------|----------------|
| **Linear Regression** | Regression | O(n) | scikit-learn |
| **Logistic Regression** | Klassifikation | O(n) | scikit-learn |
| **k-Nearest Neighbors** | Klassifikation | O(n·d) pro Vorhersage | ✅ NumPy |
| **Decision Tree** | Klassifikation | O(n·d·depth) | ✅ NumPy |

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Die beste Art, ML-Algorithmen zu verstehen, ist, sie selbst zu implementieren. Dieses Projekt enthält saubere, kommentierte Eigenimplementierungen von k-NN und Decision Tree.*
