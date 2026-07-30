"""
W&B Experiment Tracking für KI-Algorithmen
==========================================
Integriert Weights & Biases in die KI-Algorithmen-Sammlung.
Loggt Metriken, Hyperparameter und Visualisierungen.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="ki-algorithmen", config={...})
    tracker.log_metrics({"accuracy": 0.95, "train_time": 1.2})
    tracker.finish()
"""

import os
import subprocess
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für ML-Algorithmen.

    Features:
    - Automatischer Offline-Modus ohne API-Key
    - Konsistente Metrik-Namen
    - Hyperparameter-Logging
    - Laufzeit-Messung
    - Tabelle für Vorhersagen
    """

    def __init__(
        self,
        project: str = "ki-algorithmen",
        config: dict | None = None,
        tags: list | None = None,
        group: str | None = None,
        job_type: str = "train",
        notes: str | None = None,
        offline: bool = False,
    ):
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
                    tags=tags or ["ml", "from-scratch"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL,
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:  # noqa: S110, BLE001
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: int | None = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_metrics(self, metrics: dict, prefix: str = ""):
        """Loggt Metriken mit optionalem Präfix."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.log(metrics)

    def log_elapsed_time(self):
        """Loggt die vergangene Trainingszeit."""
        elapsed = time.time() - self._start_time
        self.log({"train_time_seconds": elapsed})

    def log_predictions_table(
        self, y_true: list, y_pred: list, class_names: list | None = None
    ):
        """Loggt eine Tabelle mit Vorhersagen."""
        if not self.run:
            return
        columns = ["index", "true", "predicted", "correct"]
        data = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            data.append([i, str(t), str(p), t == p])
        table = wandb.Table(columns=columns, data=data)
        self.run.log({"predictions": table})

    def log_confusion_matrix(self, y_true: list, y_pred: list, class_names: list):
        """Loggt eine Confusion Matrix."""
        if not self.run:
            return
        try:
            self.run.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names,
                )
            })
        except Exception:  # noqa: S110, BLE001
            pass

    def finish(self):
        """Beendet den W&B-Run."""
        self.log_elapsed_time()
        if self.run:
            self.run.finish()

    @property
    def is_active(self) -> bool:
        return self.run is not None
