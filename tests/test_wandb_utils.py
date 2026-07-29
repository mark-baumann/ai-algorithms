"""
Tests für wandb_utils.py — W&B Experiment Tracking für KI-Algorithmen.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wandb_utils import WandBTracker, WANDB_AVAILABLE


class TestWandBTracker:
    """Tests für den WandBTracker."""

    def test_initialization_offline(self):
        """Tracker sollte im Offline-Modus initialisieren."""
        tracker = WandBTracker(
            project="test-ki-algo",
            config={"algo": "knn", "k": 3},
            tags=["test"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        if WANDB_AVAILABLE:
            assert tracker.is_active
            assert tracker.run is not None
        else:
            assert not tracker.is_active
        tracker.finish()

    def test_log_metrics(self):
        """Metriken sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-ki-algo", offline=True)
        if tracker.is_active:
            tracker.log({"accuracy": 0.95})
            tracker.log_metrics({"precision": 0.92, "recall": 0.88}, prefix="test")
        tracker.finish()

    def test_log_elapsed_time(self):
        """Laufzeit sollte geloggt werden."""
        tracker = WandBTracker(project="test-ki-algo", offline=True)
        if tracker.is_active:
            tracker.log_elapsed_time()
        tracker.finish()

    def test_log_predictions_table(self):
        """Vorhersage-Tabelle sollte ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-ki-algo", offline=True)
        if tracker.is_active:
            tracker.log_predictions_table(
                y_true=[0, 1, 0, 1],
                y_pred=[0, 1, 1, 1],
            )
        tracker.finish()

    def test_log_confusion_matrix(self):
        """Confusion Matrix sollte ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-ki-algo", offline=True)
        if tracker.is_active:
            tracker.log_confusion_matrix(
                y_true=[0, 1, 0, 1],
                y_pred=[0, 1, 1, 1],
                class_names=["Klasse 0", "Klasse 1"],
            )
        tracker.finish()

    def test_finish_cleans_up(self):
        """finish() sollte den Run beenden und safe für doppelte Aufrufe sein."""
        tracker = WandBTracker(project="test-ki-algo", offline=True)
        tracker.finish()
        tracker.finish()  # Doppeltes finish() sollte safe sein

    def test_with_config(self):
        """Konfiguration sollte korrekt übergeben werden."""
        config = {"algo": "decision_tree", "max_depth": 5, "criterion": "gini"}
        tracker = WandBTracker(project="test-ki-algo", config=config, offline=True)
        if tracker.is_active:
            # Config sollte im Run gespeichert sein
            assert tracker.run.config is not None
        tracker.finish()
