"""
W&B Experiment Tracking für KI-Algorithmen
==========================================
Integriert Weights & Biases in KI-Algorithmen (KNN, Decision Tree, etc.).
Loggt Metriken, Confusion Matrices, Vorhersage-Tabellen und Laufzeiten.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="ki-algorithmen", config={...})
    tracker.log_metrics({"precision": 0.92, "recall": 0.88}, prefix="test")
    tracker.finish()
"""

import os
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für KI-Algorithmen.

    Features:
    - Metriken mit Prefix loggen
    - Confusion Matrix
    - Vorhersage-Tabellen
    - Laufzeit-Messung
    """

    def __init__(self, project: str = "ki-algorithmen",
                 config: dict = None, tags: list = None,
                 group: str = None, job_type: str = "train",
                 notes: str = None, offline: bool = False):
        self.project = project
        self.run = None
        self._start_time = time.time()

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["ki", "ml"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        import subprocess
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: int = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_metrics(self, metrics: dict, prefix: str = None):
        """Loggt Metriken mit optionalem Prefix."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.log(metrics)

    def log_elapsed_time(self):
        """Loggt die vergangene Laufzeit seit Initialisierung."""
        elapsed = time.time() - self._start_time
        self.log({"elapsed_time_seconds": elapsed})

    def log_predictions_table(self, y_true: list, y_pred: list):
        """Loggt eine Tabelle mit Vorhersagen."""
        if not self.run:
            return
        table = wandb.Table(columns=["y_true", "y_pred"])
        for t, p in zip(y_true, y_pred):
            table.add_data(t, p)
        self.run.log({"predictions": table})

    def log_confusion_matrix(self, y_true: list, y_pred: list,
                             class_names: list = None):
        """Loggt eine Confusion Matrix."""
        if not self.run:
            return
        try:
            cm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
            )
            self.run.log({"confusion_matrix": cm})
        except Exception:
            pass

    def finish(self):
        """Beendet den W&B-Run. Sicher bei mehrfachem Aufruf."""
        if self.run:
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None
