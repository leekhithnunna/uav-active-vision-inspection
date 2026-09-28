"""Live real-image detector: DeepCrack TEST patches -> U-Net with MC-dropout at view qualities q0..q4.

The U-Net definition is read from uav_inspection/scripts/build_detector_cache.py with `ast` (that script
parses arguments at import time, so it cannot be imported) - the demo never keeps its own copy of the model.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import paths

_NEEDED = {"PS", "RES", "Block", "SmallUNet", "degrade", "to_tensor"}


@lru_cache(maxsize=1)
def unet_namespace() -> dict:
    """Execute only the constant/class/function definitions we need from the cache builder."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    src_path = paths.detector_path("unet_source")
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in _NEEDED:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in _NEEDED for t in node.targets):
            keep.append(node)
    ns = {"np": np, "torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(src_path), "exec"), ns)
    missing = _NEEDED - set(ns)
    assert not missing, f"could not find {missing} in {src_path}"
    return ns


@lru_cache(maxsize=1)
def load_unet():
    """SmallUNet with trained weights; batch-norm in eval mode, dropout ON (MC-dropout)."""
    import torch
    ns = unet_namespace()
    model = ns["SmallUNet"]()
    model.load_state_dict(torch.load(paths.detector_path("unet"), map_location="cpu"))
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train()
    return model


def dataset_dir() -> Path:
    return paths.detector_path("deepcrack_dir")


def dataset_available() -> bool:
    d = dataset_dir()
    return (d / "test_img").is_dir() and (d / "test_lab").is_dir() and any((d / "test_img").glob("*.jpg"))


@dataclass
class Patch:
    image: np.ndarray       # (64, 64, 3) uint8
    label: np.ndarray       # (64, 64) uint8 ground-truth crack mask
    source: str             # file name + offset
    is_crack: bool


def test_patches(seed: int, n_crack: int = 2, n_clean: int = 1) -> list[Patch]:
    """Deterministically pick crack / clean 64x64 patches from the DeepCrack TEST split (no training images)."""
    from PIL import Image
    ps = unet_namespace()["PS"]
    d = dataset_dir()
    files = sorted((d / "test_img").glob("*.jpg"))
    rng = np.random.default_rng(seed)
    crack, clean = [], []
    for f in rng.permutation(files):
        lab_f = d / "test_lab" / (f.stem + ".png")
        if not lab_f.exists():
            continue
        img = np.array(Image.open(f).convert("RGB"), np.uint8)
        lab = (np.array(Image.open(lab_f).convert("L")) > 127).astype(np.uint8)
        cands = [(y, x) for y in range(0, lab.shape[0] - ps + 1, ps) for x in range(0, lab.shape[1] - ps + 1, ps)]
        for y, x in rng.permutation(cands):
            frac = lab[y:y + ps, x:x + ps].mean()
            p = Patch(img[y:y + ps, x:x + ps], lab[y:y + ps, x:x + ps], f"{f.name} @({y},{x})", frac > 0.01)
            if frac > 0.03 and len(crack) < n_crack:          # clearly visible crack (cache uses > 1%)
                crack.append(p)
                break
            if frac == 0 and len(clean) < n_clean and len(crack) >= n_crack:
                clean.append(p)
                break
        if len(crack) >= n_crack and len(clean) >= n_clean:
            break
    return crack + clean


@dataclass
class QView:
    q: int
    res: int
    view: np.ndarray        # degraded image (64, 64, 3) float 0..1
    mean: np.ndarray        # MC-dropout mean crack probability (64, 64)
    std: np.ndarray         # MC-dropout std (epistemic) (64, 64)
    entropy: float          # mean binary predictive entropy (bits) - the cache's raw u
    iou: float
    detected: bool          # > 0.5 % of pixels predicted as crack (cache rule)


def analyse(patch: Patch, mc: int, seed: int = 0) -> list[QView]:
    """Run T MC-dropout passes at every view quality q = 0..4, exactly like build_detector_cache.py."""
    import torch
    ns, model = unet_namespace(), load_unet()
    torch.manual_seed(seed)
    x0 = ns["to_tensor"](patch.image[None])
    gt = torch.from_numpy(patch.label).bool()
    out = []
    with torch.no_grad():
        for q, res in enumerate(ns["RES"]):
            x = ns["degrade"](x0, q)
            probs = torch.stack([torch.sigmoid(model(x)) for _ in range(mc)])[:, 0, 0]
            mean, std = probs.mean(0), probs.std(0)
            pm = mean.clamp(1e-6, 1 - 1e-6)
            ent = (-(pm * pm.log() + (1 - pm) * (1 - pm).log()) / np.log(2)).mean().item()
            pred = mean > 0.5
            union = (pred | gt).sum().item()
            iou = (pred & gt).sum().item() / union if union else 1.0
            out.append(QView(q, res, x[0].permute(1, 2, 0).numpy().clip(0, 1), mean.numpy(), std.numpy(), ent, iou,
                             bool(pred.float().mean().item() > 0.005)))
    return out


def report() -> dict:
    return json.loads(paths.detector_path("report").read_text())
