"""
Streamlit-App: KI-Algorithmen — Interaktiv
===========================================
Linear/Logistic Regression interaktiv, Decision Boundary plotten.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_utils import plot_decision_boundary
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

st.set_page_config(page_title="KI-Algorithmen", page_icon="🤖", layout="wide")
st.title("🤖 KI-Algorithmen — Interaktiv")
st.markdown("Lineare & Logistische Regression, Decision Boundaries — die Klassiker des Machine Learning")

page = st.sidebar.radio(
    "Bereich wählen",
    ["Lineare Regression", "Logistische Regression", "Decision Boundary", "k-NN & Decision Tree"]
)

# ═══════════════════════════════════════════════════════════════════════════
# LINEARE REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
if page == "Lineare Regression":
    st.header("📈 Lineare Regression")

    st.markdown("""
    **Lineare Regression** modelliert die Beziehung zwischen einer abhängigen Variable y
    und einer oder mehreren unabhängigen Variablen X.
    
    **Modell:** y = w·x + b
    
    **Ziel:** Minimiere den Mean Squared Error (MSE)
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.slider("Anzahl Datenpunkte", 20, 200, 100)
    with col2:
        noise = st.slider("Rauschen", 0.0, 50.0, 20.0)
    with col3:
        n_features = st.selectbox("Features", [1, 2, 3], index=0)

    if st.button("Daten generieren & Regression", type="primary"):
        X, y, coef = make_regression(
            n_samples=n_samples, n_features=n_features, noise=noise,
            coef=True, random_state=42
        )

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("MSE (Mean Squared Error)", f"{mse:.2f}")
        with col2:
            st.metric("R² Score", f"{r2:.3f}")

        if n_features == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X[:, 0], y, alpha=0.6, label="Daten", color="#4ECDC4")

            x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
            y_line = model.predict(x_line)
            ax.plot(x_line, y_line, "r-", linewidth=2, label=f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

            ax.set_xlabel("Feature")
            ax.set_ylabel("Target")
            ax.set_title("Lineare Regression (1D)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.subheader("Modell-Parameter")
        st.write(f"**Gewichte (coef_):** {model.coef_}")
        st.write(f"**Bias (intercept_):** {model.intercept_:.3f}")

        st.info("💡 R² = 1 bedeutet perfekte Vorhersage, R² = 0 bedeutet so gut wie der Mittelwert.")

# ═══════════════════════════════════════════════════════════════════════════
# LOGISTISCHE REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Logistische Regression":
    st.header("📊 Logistische Regression")

    st.markdown("""
    **Logistische Regression** ist ein Klassifikationsalgorithmus für binäre Probleme.
    
    **Modell:** P(y=1|x) = σ(w·x + b), wobei σ die Sigmoid-Funktion ist.
    
    **Entscheidungsgrenze:** w·x + b = 0
    """)

    col1, col2 = st.columns(2)
    with col1:
        n_samples_lr = st.slider("Anzahl Datenpunkte", 50, 300, 150, key="lr_n")
    with col2:
        C = st.selectbox("Regularisierung C", [0.01, 0.1, 1.0, 10.0, 100.0], index=2,
                         help="Kleineres C = stärkere Regularisierung")

    if st.button("Daten generieren & Klassifizieren", type="primary"):
        X, y = make_classification(
            n_samples=n_samples_lr, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.metric("Test-Accuracy", f"{acc:.3f}")

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_decision_boundary(model, X_train, y_train,
                              f"Logistische Regression (C={C}, Acc={acc:.2%})", ax)
        st.pyplot(fig)

        st.subheader("Sigmoid-Funktion visualisieren")
        x_sig = np.linspace(-10, 10, 200)
        y_sig = 1 / (1 + np.exp(-x_sig))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_sig, y_sig, linewidth=2, color="#FF6B6B")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Schwelle (0.5)")
        ax.set_xlabel("w·x + b (Logit)")
        ax.set_ylabel("P(y=1|x)")
        ax.set_title("Sigmoid: Von Logits zu Wahrscheinlichkeiten")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.info("💡 Die Entscheidungsgrenze liegt bei P(y=1|x) = 0.5, also w·x + b = 0.")

# ═══════════════════════════════════════════════════════════════════════════
# DECISION BOUNDARY
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Decision Boundary":
    st.header("🎯 Decision Boundary — Interaktiv")

    st.markdown("""
    Die **Decision Boundary** ist die Grenze, an der sich ein Klassifikator zwischen zwei Klassen entscheidet.
    
    Verschiedene Algorithmen erzeugen unterschiedliche Grenzen:
    - **Linear:** Gerade Linie (Logistische Regression, lineare SVM)
    - **Nicht-linear:** Kurven, Kreise (k-NN, Decision Trees, Neuronale Netze)
    """)

    dataset_type = st.selectbox(
        "Datensatz-Typ",
        ["Linear trennbar", "Halbmonde (moons)", "Kreise (circles)"]
    )

    if dataset_type == "Linear trennbar":
        X, y = make_classification(
            n_samples=200, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, random_state=42
        )
    elif dataset_type == "Halbmonde (moons)":
        X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    else:
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    st.subheader("Logistische Regression")
    C_val = st.slider("Regularisierung C", 0.01, 100.0, 1.0, key="db_c")

    lr_model = LogisticRegression(C=C_val, max_iter=2000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_decision_boundary(lr_model, X_train, y_train,
                          f"Logistische Regression (C={C_val}, Acc={lr_acc:.2%})", ax)
    st.pyplot(fig)

    st.markdown(f"""
    **Ergebnis:**
    - Accuracy: {lr_acc:.3f}
    - Die Decision Boundary ist **linear** — eine gerade Linie (bzw. Hyperplane)
    - Bei nicht-linear trennbaren Daten (Moons, Circles) stößt die logistische Regression an ihre Grenzen!
    """)

# ═══════════════════════════════════════════════════════════════════════════
# k-NN & DECISION TREE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "k-NN & Decision Tree":
    st.header("🌳 k-NN & Decision Tree — Vergleich")

    st.markdown("""
    **k-NN** und **Decision Trees** sind nicht-lineare Klassifikatoren,
    die auch komplexe Entscheidungsgrenzen lernen können.
    """)

    from knn_from_scratch import KNearestNeighbors
    from decision_tree import DecisionTree

    dataset_type2 = st.selectbox(
        "Datensatz-Typ",
        ["Linear trennbar", "Halbmonde (moons)", "Kreise (circles)"],
        key="dt2"
    )

    if dataset_type2 == "Linear trennbar":
        X, y = make_classification(
            n_samples=200, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, random_state=42
        )
    elif dataset_type2 == "Halbmonde (moons)":
        X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    else:
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("k-NN")
        k = st.slider("k (Nachbarn)", 1, 21, 5, step=2, key="knn_k")
        metric = st.selectbox("Distanzmetrik", ["euclidean", "manhattan"], key="knn_metric")

        knn = KNearestNeighbors(k=k, metric=metric)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_acc = accuracy_score(y_test, knn_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_decision_boundary(knn, X_train, y_train,
                              f"k-NN (k={k}, {metric}, Acc={knn_acc:.2%})", ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Decision Tree")
        max_depth = st.slider("max_depth", 1, 20, 4, key="dt_depth")
        criterion = st.selectbox("Kriterium", ["gini", "entropy"], key="dt_crit")

        dt = DecisionTree(max_depth=max_depth, criterion=criterion)
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_decision_boundary(dt, X_train, y_train,
                              f"Decision Tree (depth={max_depth}, {criterion}, Acc={dt_acc:.2%})", ax)
        st.pyplot(fig)

    st.subheader("Vergleich")
    col1, col2, col3 = st.columns(3)
    with col1:
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train, y_train)
        lr_acc = accuracy_score(y_test, lr.predict(X_test))
        st.metric("Logistische Regression", f"{lr_acc:.3f}")
    with col2:
        st.metric("k-NN", f"{knn_acc:.3f}")
    with col3:
        st.metric("Decision Tree", f"{dt_acc:.3f}")

    st.info("""
    💡 **Erkenntnis:**
    - **Lineare Modelle** (Logistische Regression) versagen bei nicht-linear trennbaren Daten
    - **k-NN** und **Decision Trees** können beliebige Entscheidungsgrenzen lernen
    - **k-NN** braucht keine Trainingsphase (lazy learner)
    - **Decision Trees** sind interpretierbar — man kann die Regeln ablesen!
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("📚 **KI-Algorithmen** — Interaktives Lernen")
st.sidebar.markdown("[GitHub Repository](https://github.com/mark-baumann/ki-algorithmen)")
