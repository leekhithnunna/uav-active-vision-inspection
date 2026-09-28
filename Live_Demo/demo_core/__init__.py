"""Reusable library for the Team 08 live demo. Importing it makes `env` (uav_inspection) importable."""
import sys

# Windows consoles default to cp1252: force UTF-8 so mu/sigma/pi/epsilon and box drawing never crash a demo.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from . import paths  # noqa: E402,F401  (side effect: uav_inspection/ on sys.path)
