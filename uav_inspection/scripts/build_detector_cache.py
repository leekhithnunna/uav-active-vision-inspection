"""
Real-image detector for the environment (Level 2): DeepCrack -> small U-Net with MC-dropout -> detector_cache.npz

Pipeline
  1. Train a small U-Net on random 64x64 crops from DeepCrack TRAIN split (GPU: ~5 min).
  2. Cut 64x64 patches from the TEST split only (no leakage): crack >1% -> defect, crack == 0 -> clean.
  3. For each patch and each view quality q=0..4 (effective res 4/8/16/32/64 px), run T MC-dropout passes:
       u    = mean per-pixel predictive entropy of the MC-dropout mean prediction
              (total uncertainty, Gal & Ghahramani 2016), rank-normalised to [0,1]
              and made non-increasing in q ("a better view never loses information")
       pdef = predicted crack pixels > 0.5% of the patch
       iou  = IoU of predicted mask vs ground truth
  4. Save detector_cache.npz in the env's cache format (+ recommended success threshold), a report and a figure.

Colab:
  !git clone -q --depth 1 https://github.com/yhlleo/DeepCrack.git /content/DeepCrack
  !unzip -q -o /content/DeepCrack/dataset/DeepCrack.zip -d /content/deepcrack_data
  !python scripts/build_detector_cache.py --data /content/deepcrack_data --out $DRIVE
"""
import argparse, glob, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="folder with train_img/ train_lab/ test_img/ test_lab/")
p.add_argument("--out", required=True, help="output folder (e.g. your Drive team08 folder)")
p.add_argument("--iters", type=int, default=4000, help="training iterations")
p.add_argument("--batch", type=int, default=32)
p.add_argument("--max_defect", type=int, default=1200, help="max defect patches in cache")
p.add_argument("--max_clean", type=int, default=1800, help="max clean patches in cache")
p.add_argument("--mc", type=int, default=10, help="MC-dropout passes")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--reuse_model", action="store_true", help="skip training, load unet_deepcrack.pt from --out")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}")

PS = 64                                  # patch size (one env cell)
RES = [4, 8, 16, 32, 64]                 # effective resolution for quality q = 0..4 (each altitude step halves detail)


# ------------------------------------------------------------------ data
def load_split(split):
    imgs, labs = [], []
    for f in sorted(glob.glob(os.path.join(args.data, f"{split}_img", "*.jpg"))):
        name = os.path.splitext(os.path.basename(f))[0]
        lab_f = os.path.join(args.data, f"{split}_lab", name + ".png")
        if not os.path.exists(lab_f):
            continue
        imgs.append(np.array(Image.open(f).convert("RGB"), np.uint8))
        labs.append((np.array(Image.open(lab_f).convert("L")) > 127).astype(np.uint8))
    return imgs, labs


def degrade(x, q):
    """x: (N,3,64,64) float tensor. Simulates a worse view: downsample to RES[q] then upsample back."""
    r = RES[q]
    if r >= PS:
        return x
    small = F.interpolate(x, size=(r, r), mode="area")
    return F.interpolate(small, size=(PS, PS), mode="bilinear", align_corners=False)


def random_crops(imgs, labs, n):
    X = np.empty((n, PS, PS, 3), np.uint8); Y = np.empty((n, PS, PS), np.uint8)
    for k in range(n):
        i = rng.integers(len(imgs)); im, lb = imgs[i], labs[i]
        H, W = lb.shape
        if rng.random() < 0.6 and lb.any():                      # 60% crops centred on a crack pixel
            ys, xs = np.nonzero(lb); j = rng.integers(len(ys))
            y0 = int(np.clip(ys[j] - PS // 2 + rng.integers(-16, 17), 0, H - PS))
            x0 = int(np.clip(xs[j] - PS // 2 + rng.integers(-16, 17), 0, W - PS))
        else:
            y0, x0 = rng.integers(0, H - PS + 1), rng.integers(0, W - PS + 1)
        x, y = im[y0:y0 + PS, x0:x0 + PS], lb[y0:y0 + PS, x0:x0 + PS]
        if rng.random() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
        if rng.random() < 0.5: x, y = x[::-1], y[::-1]
        X[k], Y[k] = x, y
    return X, Y


def to_tensor(X):
    return torch.from_numpy(X).permute(0, 3, 1, 2).float().div(255.0)


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, cin, cout, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity())

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    """4-level U-Net (16-32-64-128) with dropout in bottleneck + decoder -> MC-dropout uncertainty."""
    def __init__(self, p=0.2):
        super().__init__()
        self.e1, self.e2, self.e3 = Block(3, 16), Block(16, 32), Block(32, 64)
        self.b = Block(64, 128, p)
        self.u3, self.d3 = nn.ConvTranspose2d(128, 64, 2, 2), Block(128, 64, p)
        self.u2, self.d2 = nn.ConvTranspose2d(64, 32, 2, 2), Block(64, 32, p)
        self.u1, self.d1 = nn.ConvTranspose2d(32, 16, 2, 2), Block(32, 16, p)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(5.0, device=y.device))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


# ------------------------------------------------------------------ 1. train on TRAIN split
t0 = time.time()
tr_imgs, tr_labs = load_split("train")
te_imgs, te_labs = load_split("test")
print(f"train images: {len(tr_imgs)}   test images: {len(te_imgs)}")
assert tr_imgs and te_imgs, "no images found - check --data points to the unzipped DeepCrack folder"

model = SmallUNet().to(dev)
ckpt = os.path.join(args.out, "unet_deepcrack.pt")
if args.reuse_model and os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    print("loaded existing U-Net, skipping training")
else:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    model.train()
    for it in range(1, args.iters + 1):
        X, Y = random_crops(tr_imgs, tr_labs, args.batch)
        x, y = to_tensor(X).to(dev), torch.from_numpy(Y).float().unsqueeze(1).to(dev)
        if rng.random() < 0.3:                                  # sometimes train on a worse view too
            x = degrade(x, int(rng.integers(1, 4)))
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % max(1, args.iters // 10) == 0:
            print(f"iter {it:5d}/{args.iters}  loss={loss.item():.4f}")
    torch.save(model.state_dict(), ckpt)
    print(f"U-Net trained in {(time.time() - t0) / 60:.1f} min")

# ------------------------------------------------------------------ 2. patches from TEST split only
P_img, P_lab = [], []
for im, lb in zip(te_imgs, te_labs):
    H, W = lb.shape
    for y0 in range(0, H - PS + 1, PS):
        for x0 in range(0, W - PS + 1, PS):
            P_img.append(im[y0:y0 + PS, x0:x0 + PS]); P_lab.append(lb[y0:y0 + PS, x0:x0 + PS])
P_img, P_lab = np.stack(P_img), np.stack(P_lab)
frac = P_lab.reshape(len(P_lab), -1).mean(1)
def_idx, clean_idx = np.where(frac > 0.01)[0], np.where(frac == 0)[0]   # 0 < frac <= 1% dropped (ambiguous)
def_idx = rng.permutation(def_idx)[:args.max_defect]
clean_idx = rng.permutation(clean_idx)[:args.max_clean]
keep = np.concatenate([def_idx, clean_idx])
P_img, P_lab = P_img[keep], P_lab[keep]
is_defect = np.concatenate([np.ones(len(def_idx), bool), np.zeros(len(clean_idx), bool)])
print(f"cache patches: {len(keep)}  (defect {len(def_idx)}, clean {len(clean_idx)})")

# ------------------------------------------------------------------ 3. MC-dropout at 5 view qualities
model.eval()
for m in model.modules():                                       # dropout ON, batch-norm stays in eval mode
    if isinstance(m, nn.Dropout2d):
        m.train()
N = len(keep)
u_raw = np.zeros((N, 5), np.float32); u_std = np.zeros((N, 5), np.float32); pdef = np.zeros((N, 5), bool); iou = np.zeros((N, 5), np.float32)
gt = torch.from_numpy(P_lab).bool()
examples = {}
with torch.no_grad():
    for q in range(5):
        for s in range(0, N, 256):
            x = degrade(to_tensor(P_img[s:s + 256]).to(dev), q)
            probs = torch.stack([torch.sigmoid(model(x)) for _ in range(args.mc)])   # (T,B,1,64,64)
            mean, std = probs.mean(0)[:, 0].cpu(), probs.std(0)[:, 0].cpu()
            pred = mean > 0.5
            g = gt[s:s + 256]
            inter = (pred & g).flatten(1).sum(1).float(); union = (pred | g).flatten(1).sum(1).float()
            pm = mean.clamp(1e-6, 1 - 1e-6)
            ent = -(pm * pm.log() + (1 - pm) * (1 - pm).log()) / np.log(2)      # binary entropy in bits
            u_raw[s:s + 256, q] = ent.flatten(1).mean(1).numpy()               # total predictive uncertainty
            u_std[s:s + 256, q] = std.flatten(1).mean(1).numpy()               # epistemic part (kept for report)
            pdef[s:s + 256, q] = (pred.flatten(1).float().mean(1) > 0.005).numpy()
            iou[s:s + 256, q] = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter)).numpy()
            if s == 0:
                examples[q] = (x[:1].cpu(), mean[:1], std[:1])
        print(f"q={q} ({RES[q]:2d}px) done")

# rank-normalise uncertainty to [0,1] across all (patch, q) values: keeps ordering, spreads the scale
flat = u_raw.ravel()
u = (np.argsort(np.argsort(flat)) / (len(flat) - 1)).reshape(u_raw.shape).astype(np.float32)
mono_violations = float(np.mean(np.diff(u, axis=1) > 0))       # share of q-steps where u went UP with a better view
u = np.minimum.accumulate(u, axis=1)                            # best-view fusion: information never hurts

# altitude -> view quality for the real detector: high=4px, mid=8px, low=32px (q=4 / 64px only via Agent B).
# The U-Net stays usable down to ~16px, so mid altitude must be coarser for descending to matter.
ALT_Q_LEARNED = [0, 1, 3]
# recommended navigate-success threshold: 75% of the way from a full low scan to a full mid scan
m = lambda q: 0.2 * u[is_defect, q].mean() + 0.8 * u[~is_defect, q].mean()
q_mid, q_low = ALT_Q_LEARNED[1], ALT_Q_LEARNED[2]
success_unc = float(m(q_low) + 0.75 * (m(q_mid) - m(q_low)))   # full mid scan fails; ~20-30 targeted descents needed (as in analytic mode)

np.savez(os.path.join(args.out, "detector_cache.npz"), u=u, pdef=pdef, iou=iou, is_defect=is_defect,
         u_raw=u_raw, u_std=u_std, success_unc=np.float32(success_unc),
         alt_q=np.array(ALT_Q_LEARNED))

# ------------------------------------------------------------------ 4. report + figure
report = []
print(f"\n{'q':>2} {'res':>4} | {'u defect':>8} {'u clean':>8} | {'detect rate':>11} {'false pos':>9} | {'IoU defect':>10}")
for q in range(5):
    row = dict(q=q, res=RES[q], u_defect=float(u[is_defect, q].mean()), u_clean=float(u[~is_defect, q].mean()),
               detect_rate=float(pdef[is_defect, q].mean()), false_pos=float(pdef[~is_defect, q].mean()),
               iou_defect=float(iou[is_defect, q].mean()))
    report.append(row)
    print(f"{q:>2} {RES[q]:>4} | {row['u_defect']:8.3f} {row['u_clean']:8.3f} | {row['detect_rate']:11.3f} "
          f"{row['false_pos']:9.3f} | {row['iou_defect']:10.3f}")
print(f"\nraw monotonicity violations (before fusion): {mono_violations:.1%}")
print(f"\nrecommended success_unc for navigate mode: {success_unc:.3f} (stored inside the cache)")
json.dump({"report": report, "success_unc": success_unc, "alt_q": ALT_Q_LEARNED, "mono_violations": mono_violations, "n_defect": int(is_defect.sum()),
           "n_clean": int((~is_defect).sum()), "train_iters": args.iters, "mc_passes": args.mc},
          open(os.path.join(args.out, "detector_report.json"), "w"), indent=2)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = int(np.argmax(is_defect))                                   # first defect patch
    fig, ax = plt.subplots(3, 5, figsize=(12, 7.5))
    with torch.no_grad():
        for q in range(5):
            x = degrade(to_tensor(P_img[k:k + 1]).to(dev), q)
            probs = torch.stack([torch.sigmoid(model(x)) for _ in range(args.mc)])
            ax[0, q].imshow(x[0].permute(1, 2, 0).cpu().numpy()); ax[0, q].set_title(f"view q={q} ({RES[q]}px)")
            ax[1, q].imshow(probs.mean(0)[0, 0].cpu(), cmap="gray", vmin=0, vmax=1); ax[1, q].set_title("mean prediction")
            ax[2, q].imshow(probs.std(0)[0, 0].cpu(), cmap="magma"); ax[2, q].set_title(f"uncertainty u={u[k, q]:.2f}")
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "detector_views.png"), dpi=130)
    print("saved detector_views.png")
except Exception as e:
    print("figure skipped:", e)
print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> {os.path.join(args.out, 'detector_cache.npz')}")
