"""Generate and execute notebooks/Live_Demo.ipynb (same demos as cells, calling demo_core; outputs kept).

    python scripts/build_notebook.py          (~5 min: it runs every demo with --save)
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Live_Demo.ipynb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell


def show(*files: str) -> str:
    """Code that displays saved GIF/PNG files under the cell."""
    return "\n".join(f"show('{f}')" for f in files)


CELLS = [
    md("# Team 08 - Uncertainty-Aware Active Vision for UAV Bridge Inspection - Live Demo\n"
       "Same demos as `run_demo.py`. Every cell calls a demo's `main()` (all logic lives in `demo_core`), "
       "prints the narrated terminal output, then shows the GIF / PNG that run just produced in `backup/`.\n\n"
       "Guideline steps: **1** start env, **2** observation, **3** actions + selection, **4** env response, "
       "**5** reward + next state, **6** learning update, **7** trained policy, **8** curves + metrics."),
    code("import sys, os\n"
         "from pathlib import Path\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT)); os.chdir(ROOT)\n"
         "from IPython.display import Image, display\n"
         "from demo_core import paths\n"
         "def show(name):\n"
         "    for kind in ('gifs', 'frames', 'snapshots'):\n"
         "        p = paths.out_dir(kind) / name\n"
         "        if p.exists():\n"
         "            return display(Image(filename=str(p)))\n"
         "    print('not found:', name)\n"
         "SEED = ['--seed', str(paths.default_seed())]"),
    md("## Demo 1 - Environment tour (guideline 1, 2, 4)"),
    code("import demos.d1_env_tour as d1\nd1.main(['--save', '--fast'] + SEED)\n" + show("env_tour_final.png")),
    md("## Demo 2 - Agent A (DQN) step-through (guideline 2-5, 7, 8)"),
    code("import demos.d2_agentA_step as d2\nd2.main(['--save', '--auto'] + SEED)\n"
         + show("agentA_step1.png", "agentA_final.png")),
    md("## Demo 3 - Agent A vs dense raster, same bridge (guideline 7, 8)"),
    code("import demos.d3_agentA_vs_raster as d3\nd3.main(['--save'] + SEED)\n" + show("agentA_vs_raster.gif")),
    md("## Demo 4 - Agent B (PPO) view refinement vs FixedOrbit (guideline 2-5, 7, 8)"),
    code("import demos.d4_agentB_refine as d4\nd4.main(['--save'] + SEED)\n" + show("agentB_refine.gif")),
    md("## Demo 5 - Real-image detector: U-Net + MC-dropout on DeepCrack TEST patches"),
    code("import demos.d5_detector_views as d5\nd5.main(['--save'] + SEED)\n" + show("detector_crack.png")),
    md("## Demo 6 - Agent C (A2C) mission supervisor (guideline 2-5, 7, 8)"),
    code("import demos.d6_agentC_mission as d6\nd6.main(['--save'] + SEED)\n" + show("agentC_mission.gif")),
    md("## Demo 7 - Full hierarchy: C -> A / B vs all-scripted (guideline 7, 8)\n"
       "The trained supervisor never picks REFINE (true on all 20 evaluation bridges), so the second run uses "
       "the rule supervisor to show Agent B inside the hierarchy."),
    code("import demos.d7_full_mission as d7\nd7.main(['--save'] + SEED)\n" + show("full_mission.gif")),
    code("d7.main(['--save', '--supervisor', 'rule'] + SEED)\n" + show("full_mission_rule.gif")),
    md("## Demo 8 - Learning burst: the update step (guideline 6)"),
    code("import demos.d8_learning_burst as d8\nd8.main(['--save', '--agent', 'A'] + SEED)\n" + show("learning_burst_A.png")),
    md("## Demo 9 - Learning curves and final metrics from the saved results (guideline 8)"),
    code("import demos.d9_results_dashboard as d9\nd9.main(['--headless'] + SEED)\n"
         + show("results_A.png", "results_B.png", "results_C.png", "results_full.png")),
]


def main() -> int:
    nb = nbformat.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.parent.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
