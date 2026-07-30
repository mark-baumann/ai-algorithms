"""
Gemeinsame pytest-Konfiguration und Fixtures für KI-Algorithmen-Tests.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # Kein GUI-Backend in Tests


# Stelle sicher, dass das Projektverzeichnis im Pfad ist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
