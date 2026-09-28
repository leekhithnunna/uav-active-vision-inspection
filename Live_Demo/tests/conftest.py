"""Headless test setup: Agg backend, Live_Demo/ importable, narration captured instead of printed."""
import io
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def quiet_console():
    from rich.console import Console
    from demo_core import narrate
    saved, narrate.console = narrate.console, Console(file=io.StringIO(), width=130)
    yield narrate.console
    narrate.console = saved
