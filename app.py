"""
Streamlit-App: KI-Algorithmen
==============================
Linear/Logistic Regression interaktiv, Decision Boundary plotten,
k-NN & Decision Tree von Grund auf.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.colors import ListedColormap
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import plot_decision_boundary
from knn_from_scratch import KNearestNeighbors
from decision_tree import DecisionTree

st.set_page_config(page_title="KI-Algorithmen", layout="wide")
st.title("🤖 KI-Algorithmen — Interaktiv")
st.markdown("### Linear/Logistic Regression, Decision Boundaries, k-NN & Decision Tree")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Linear Regression", "📊 Logistic Regression", "🔍 k-Nearest Neighbors", "🌳 Decision Tree"
])

# ═══════════════════════════════════════════════════════════════
# Tab 1: Linear Regression
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("📈 Linear Regression — interaktiv")

    st.markdown(r"""
    **Lineare Regression** modelliert den Zusammenhang zwischen einer abhängigen Variable $y$
    und einer oder mehreren unabhängigen Variablen $x$:

    **Modell:** $y = w \cdot x + b$

    - $w$: Steigung (Gewicht)
    - $b$: Achsenabschnitt (Bias)
    """)

    col_lr1, col_lr2 = st.columns([1, 2])

    with col_lr1:
        n_points = st.slider("Anzahl Datenpunkte", 20, 200, 80)
        noise = st.slider("Rauschen", 0.0, 5.0, 2.0)
        true_w = st.slider("Wahre Steigung", -5.0, 5.0, 2.5, 0.1)
        true_b = st.slider("Wahrer Achsenabschnitt", -10.0, 10.0, 1.0, 0.5)

    with col_lr2:
        # Generiere Daten
        np.random.seed(42)
        X_lr = np.random.uniform(-5, 5, n_points).reshape(-1, 1)
        y_lr = true_w * X_lr.ravel() + true_b + np.random.randn(n_points) * noise

        # Modell fitten
        model_lr = LinearRegression()
        model_lr.fit(X_lr, y_lr)
        y_pred_lr = model_lr.predict(X_lr)

        w_learned = model_lr.coef_[0]
        b_learned = model_lr.intercept_

        # Plot
        fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
        ax_lr.scatter(X_lr, y_lr, alpha=0.6, s=40, label='Datenpunkte', c='#4A90D9')
        x_line = np.linspace(-5, 5, 100).reshape(-1, 1)
        ax_lr.plot(x_line, true_w * x_line.ravel() + true_b, 'g--', linewidth=2,
                   label=f'Wahre Linie: y = {true_w:.1f}x + {true_b:.1f}')
        ax_lr.plot(x_line, model_lr.predict(x_line), 'r-', linewidth=2,
                   label=f'Gelernte Linie: y = {w_learned:.2f}x + {b_learned:.2f}')
        ax_lr.set_xlabel('x')
        ax_lr.set_ylabel('y')
        ax_lr.set_title('Lineare Regression', fontweight='bold')
        ax_lr.legend()
        ax_lr.grid(True, alpha=0.3)
        st.pyplot(fig_lr)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Gelerntes w", f"{w_learned:.3f}", f"Wahr: {true_w:.1f}")
        col_m2.metric("Gelerntes b", f"{b_learned:.3f}", f"Wahr: {true_b:.1f}")
        col_m3.metric("R² Score", f"{model_lr.score(X_lr, y_lr):.3f}")

    st.subheader("📖 Die Mathematik dahinter")
    st.markdown(r"""
    **Ziel:** Minimiere den Mean Squared Error (MSE):

    $$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

    **Lösung (Normal Equation):**

    $$w = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}, \quad b = \bar{y} - w\bar{x}$$

    Oder per **Gradient Descent** (siehe Mathe-Algorithmen-App)!
    """)

# ═══════════════════════════════════════════════════════════════
# Tab 2: Logistic Regression
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Logistic Regression — Decision Boundary")

    st.markdown(r"""
    **Logistic Regression** ist ein Klassifikations-Algorithmus (trotz des Namens!).
    Sie modelliert die Wahrscheinlichkeit, dass ein Punkt zu Klasse 1 gehört:

    $$P(y=1|x) = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

    Die **Decision Boundary** ist die Linie, bei der $P(y=1|x) = 0.5$.
    """)

    col_log1, col_log2 = st.columns([1, 2])

    with col_log1:
        dataset_type = st.selectbox("Datensatz-Typ", [
            "Linear trennbar", "Halbmonde (Moons)", "Kreise", "Blobs"
        ])
        C_reg = st.select_slider(
            "Regularisierung C (invers)",
            options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
            value=1.0
        )
        st.caption("Kleines C = starke Regularisierung")

    with col_log2:
        # Datensatz generieren
        np.random.seed(42)
        if dataset_type == "Linear trennbar":
            X_log, y_log = make_classification(
                n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                n_clusters_per_class=1, random_state=42
            )
        elif dataset_type == "Halbmonde (Moons)":
            X_log, y_log = make_moons(n_samples=200, noise=0.2, random_state=42)
        elif dataset_type == "Kreise":
            X_log, y_log = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
        else:
            X_log, y_log = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42)

        X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
            X_log, y_log, test_size=0.3, random_state=42
        )

        # Modell trainieren
        model_log = LogisticRegression(C=C_reg, max_iter=1000)
        model_log.fit(X_train_log, y_train_log)
        y_pred_log = model_log.predict(X_test_log)
        acc_log = accuracy_score(y_test_log, y_pred_log)

        # Decision Boundary plotten
        fig_log, ax_log = plt.subplots(figsize=(8, 7))
        plot_decision_boundary(model_log, X_train_log, y_train_log,
                               f'Logistic Regression (C={C_reg})\nTest-Accuracy: {acc_log:.2%}',
                               ax_log)
        st.pyplot(fig_log)

        col_lm1, col_lm2, col_lm3 = st.columns(3)
        col_lm1.metric("Test-Accuracy", f"{acc_log:.2%}")
        col_lm2.metric("Trainingsdaten", f"{len(X_train_log)}")
        col_lm3.metric("Features", "2")

    st.subheader("📖 Decision Boundary verstehen")
    st.markdown(r"""
    Die **Decision Boundary** ist die Linie (oder Kurve), die die Klassen trennt:

    - **Lineare Decision Boundary:** $w_1 x_1 + w_2 x_2 + b = 0$
    - Punkte auf der einen Seite → Klasse 0
    - Punkte auf der anderen Seite → Klasse 1

    **Logistic Regression** kann nur **lineare** Grenzen lernen.
    Für nicht-lineare Grenzen braucht man:
    - Polynomial Features
    - Kernel SVM
    - Neuronale Netze
    - Entscheidungsbäume
    """)

    # Zeige Gewichte
    st.subheader("Gelernte Gewichte")
    w_log = model_log.coef_[0]
    b_log = model_log.intercept_[0]
    st.markdown(f"""
    - **w₁** (Feature 1): {w_log[0]:.4f}
    - **w₂** (Feature 2): {w_log[1]:.4f}
    - **b** (Bias): {b_log:.4f}
    - **Decision Boundary:** {w_log[0]:.2f}·x₁ + {w_log[1]:.2f}·x₂ + {b_log:.2f} = 0
    """)

# ═══════════════════════════════════════════════════════════════
# Tab 3: k-Nearest Neighbors
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("🔍 k-Nearest Neighbors — von Grund auf")

    st.markdown("""
    **k-NN** ist einer der einfachsten ML-Algorithmen:
    1. Für einen neuen Punkt: Finde die k nächsten Nachbarn
    2. Mehrheitsentscheid der Nachbarn → Klasse

    k-NN ist ein **"lazy learner"** — kein Training, nur Speichern der Daten!
    """)

    col_knn1, col_knn2 = st.columns([1, 2])

    with col_knn1:
        k_value = st.slider("k (Anzahl Nachbarn)", 1, 31, 5, step=2)
        metric_knn = st.selectbox("Distanzmetrik", ["euclidean", "manhattan"])
        dataset_knn = st.selectbox("Datensatz", [
            "Linear trennbar", "Halbmonde", "Kreise"
        ], key="knn_dataset")

    with col_knn2:
        np.random.seed(42)
        if dataset_knn == "Linear trennbar":
            X_knn, y_knn = make_classification(
                n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                n_clusters_per_class=1, random_state=42
            )
        elif dataset_knn == "Halbmonde":
            X_knn, y_knn = make_moons(n_samples=200, noise=0.2, random_state=42)
        else:
            X_knn, y_knn = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

        X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
            X_knn, y_knn, test_size=0.3, random_state=42
        )

        model_knn = KNearestNeighbors(k=k_value, metric=metric_knn)
        model_knn.fit(X_train_knn, y_train_knn)
        y_pred_knn = model_knn.predict(X_test_knn)
        acc_knn = accuracy_score(y_test_knn, y_pred_knn)

        fig_knn, ax_knn = plt.subplots(figsize=(8, 7))
        plot_decision_boundary(
            model_knn, X_train_knn, y_train_knn,
            f'k-NN (k={k_value}, {metric_knn})\nTest-Accuracy: {acc_knn:.2%}',
            ax_knn
        )
        st.pyplot(fig_knn)

        col_km1, col_km2 = st.columns(2)
        col_km1.metric("Test-Accuracy", f"{acc_knn:.2%}")
        col_km2.metric("k", k_value)

    st.subheader("📖 Einfluss von k")
    st.markdown("""
    | k | Entscheidungsgrenze | Over/Underfitting |
    |---|-------------------|-------------------|
    | k=1 | Sehr zackig, passt sich jedem Punkt an | **Overfitting** |
    | k=5 | Ausgewogen, gute Generalisierung | ✅ Optimal |
    | k=21 | Sehr glatt, ignoriert feine Strukturen | **Underfitting** |

    **Faustregel:** $k \approx \sqrt{n}$ (Quadratwurzel der Trainingsdaten)
    """)

# ═══════════════════════════════════════════════════════════════
# Tab 4: Decision Tree
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("🌳 Decision Tree — von Grund auf")

    st.markdown("""
    Ein **Entscheidungsbaum** teilt die Daten rekursiv anhand von Features auf.
    Jeder Split maximiert den **Information Gain** — die Reduktion der Unreinheit.

    **Vorteil:** Interpretierbar! Man kann die Regeln direkt ablesen.
    """)

    col_dt1, col_dt2 = st.columns([1, 2])

    with col_dt1:
        max_depth = st.slider("max_depth", 1, 20, 4)
        criterion = st.selectbox("Split-Kriterium", ["gini", "entropy"])
        dataset_dt = st.selectbox("Datensatz", [
            "Linear trennbar", "Halbmonde", "Kreise"
        ], key="dt_dataset")

    with col_dt2:
        np.random.seed(42)
        if dataset_dt == "Linear trennbar":
            X_dt, y_dt = make_classification(
                n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                n_clusters_per_class=1, random_state=42
            )
        elif dataset_dt == "Halbmonde":
            X_dt, y_dt = make_moons(n_samples=200, noise=0.2, random_state=42)
        else:
            X_dt, y_dt = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

        X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
            X_dt, y_dt, test_size=0.3, random_state=42
        )

        model_dt = DecisionTree(max_depth=max_depth, criterion=criterion)
        model_dt.fit(X_train_dt, y_train_dt)
        y_pred_dt = model_dt.predict(X_test_dt)
        acc_dt = accuracy_score(y_test_dt, y_pred_dt)

        fig_dt, ax_dt = plt.subplots(figsize=(8, 7))
        plot_decision_boundary(
            model_dt, X_train_dt, y_train_dt,
            f'Decision Tree (max_depth={max_depth}, {criterion})\nTest-Accuracy: {acc_dt:.2%}',
            ax_dt
        )
        st.pyplot(fig_dt)

        col_dm1, col_dm2, col_dm3 = st.columns(3)
        col_dm1.metric("Test-Accuracy", f"{acc_dt:.2%}")
        col_dm2.metric("max_depth", max_depth)
        col_dm3.metric("Kriterium", criterion)

    st.subheader("📖 Baumstruktur")
    if st.button("🌳 Baum anzeigen"):
        # Capture print output
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model_dt.print_tree()
        sys.stdout = old_stdout
        tree_str = buffer.getvalue()
        st.code(tree_str, language="text")

    st.subheader("📖 Gini vs. Entropy")
    st.markdown(r"""
    **Gini-Impurity:** $G = 1 - \sum p_i^2$
    - Schneller zu berechnen (kein Logarithmus)
    - Tendiert dazu, die häufigste Klasse zu isolieren

    **Entropy:** $H = -\sum p_i \cdot \log_2(p_i)$
    - Theoretisch fundierter (Informationstheorie)
    - Tendiert zu ausgewogeneren Splits

    In der Praxis sind die Unterschiede **meist gering**.
    """)

    st.subheader("📖 Overfitting bei Entscheidungsbäumen")
    st.markdown("""
    Entscheidungsbäume **neigen stark zu Overfitting** — sie lernen das Rauschen!

    **Gegenmaßnahmen:**
    1. **max_depth begrenzen** ← wichtigster Parameter
    2. **min_samples_split** erhöhen (mind. Samples pro Split)
    3. **Pruning** (nachträglich Äste entfernen)
    4. **Random Forest** (viele Bäume → Ensemble)

    **Experiment:** Erhöhe max_depth auf 20 → Overfitting (zackige Grenzen)!
    """)

st.sidebar.markdown("""
### 📚 Über diese App

Interaktive Exploration von **KI-Algorithmen** —
von Grund auf implementiert.

**Funktionen:**
- 📈 Linear Regression mit echten Daten
- 📊 Logistic Regression & Decision Boundaries
- 🔍 k-NN mit verschiedenen k & Metriken
- 🌳 Decision Tree mit Gini/Entropy

**Code:** `knn_from_scratch.py`, `decision_tree.py`, `plot_utils.py`
""")
