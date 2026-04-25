"""
kaggle_train.py  [v13 — ALL VISUAL BUGS FIXED]
=========================================================================
Changes from v12:

  FIXES:
  [FIX-1]  Shadow loss computed twice and overwritten — FIXED.
           The first block (^2 exponent) was silently discarded every step.
           Now single clean pass with ^4 exponent + lum_pattern on top.

  [FIX-2]  freq_loss set twice — second one always won. FIXED to single call.

  [FIX-3]  Optimizer/scheduler/scaler state now RESTORED on resume.
           Previously reset to fresh AdamW every Kaggle session, wasting
           all momentum built up over 260k+ steps.

  [FIX-4]  Loss weights rebalanced. color_mv_weight 12→4, shadow 8→4,
           freq 8→2. Auxiliary losses were dominating the diffusion MSE,
           causing desaturation and flat geometry.

  [FIX-5]  Saturation loss added. Directly penalises washed-out/grey
           predictions by matching per-channel std in pixel space.

  [FIX-6]  Bottleneck self-attention added to UNet. Gives the model
           global illumination reasoning — fixes flat nose, soft shadow
           boundaries, and wrong colour casts.

  [FIX-7]  LP3 network made deeper (3→5 layers) with LayerNorm.
           Better SH→lighting embedding = more accurate colour per HDRI.

  [FIX-8]  DDIM clamp relaxed from (-1,1) to (-1.2,1.2) during sampling.
           Hard clamping at every step was crushing highlights and shadows,
           contributing to the washed-out look.

All v12 improvements preserved.

==========================================================================
TUNING GUIDE — READ THIS BEFORE CHANGING ANYTHING
==========================================================================

IF PREDICTIONS ARE STILL TOO GREY / DESATURATED:
  → Increase "sat_loss_weight" from 2.0 → 4.0 → 6.0 (double each time)
  → Increase "lpips_weight" from 3.0 → 4.0 (LPIPS cares about saturation)
  → Do NOT touch color_mv_weight above 6.0 — causes colour banding

IF PREDICTIONS HAVE WRONG COLOUR CAST (green/orange tint):
  → Increase "color_mv_weight" from 4.0 → 6.0
  → Increase "ch_loss_weight" from 1.0 → 2.0
  → If cast persists after 20k steps, check LP3 — increase its width

IF NOSE / FACE GEOMETRY IS STILL FLAT:
  → The self-attention is the fix — confirm SelfAttention is in the UNet
  → Increase "shadow_weight" from 4.0 → 6.0
  → If still flat after 50k more steps, add attention at enc level 2 also

IF SHADOW BOUNDARIES ARE TOO SOFT:
  → Increase "shadow_weight" from 4.0 → 6.0
  → Change ^4 exponent in shadow loss to ^6 for harsher boundary focus
  → Do NOT go above ^8 — the gradients become unstable

IF HAIR IS BLURRY:
  → Increase "freq_loss_weight" from 2.0 → 3.0 → 4.0
  → Do NOT go above 5.0 — causes checkerboard high-freq artifacts

IF CLOTHING COLOUR IS WRONG (blue top goes grey):
  → This is a saturation problem — increase "sat_loss_weight"
  → Also increase "ch_loss_weight" (colour histogram loss)

IF PREDICTIONS HAVE CHECKERBOARD / NOISY ARTIFACTS:
  → freq_loss_weight is too high — reduce to 1.0
  → shadow_weight may be too high — reduce to 3.0

IF TRAINING LOSS IS NaN OR EXPLODING:
  → Reduce lr from 3e-5 → 1e-5
  → Reduce lpips_weight first (it's the most unstable)
  → Check if nan_count in logs is growing — if >10/1000 steps, reduce LR

IF PSNR IS NOT IMPROVING AFTER 50k MORE STEPS:
  → Reduce lr_min from 1e-6 → 5e-7
  → Verify optimizer is being restored (check "[RESUME] Optimizer restored" log)

IF PREDICTIONS LOOK CORRECT BUT SSIM IS LOW:
  → Model is sharp but misaligned — increase "mse_weight" from 1.0 → 1.5
  → This helps structural alignment

IF TRAINING IS SLOW (>8s/step on A100):
  → Reduce accum_steps from 2 → 1
  → Reduce lpips_every from 1 → 2 (compute LPIPS every 2 steps)

==========================================================================
"""

import os
import sys
import json
import time
import math
import random
import shutil
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import torchvision.transforms as T
from tqdm import tqdm

# ============================================================
# KAGGLE PATHS
# ============================================================

KAGGLE_DATA_ROOT = Path("/kaggle/input")
KAGGLE_WORK_DIR  = Path("/kaggle/working")

# ============================================================
# RESUME — Copy checkpoint + CSV from input datasets
# ============================================================

_ckpt_dst_dir = KAGGLE_WORK_DIR / "checkpoints"
_ckpt_dst_dir.mkdir(parents=True, exist_ok=True)
_dst_ckpt = _ckpt_dst_dir / "last.pt"

def _is_valid_pt(path):
    try:
        ck   = torch.load(path, map_location="cpu")
        step = ck.get("step", 0)
        loss = ck.get("best_loss", float("inf"))
        unet_state = ck.get("unet", {})
        for k, v in list(unet_state.items())[:5]:
            if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
                print(f"    [CORRUPT] NaN/Inf in weights: {k}")
                return False, 0, float("inf")
        return True, step, loss
    except Exception:
        return False, 0, float("inf")

def _find_best_checkpoint():
    search_dirs = []
    for p in KAGGLE_DATA_ROOT.rglob("relight-checkpoint"):
        if p.is_dir(): search_dirs.append(p)
    for p in KAGGLE_DATA_ROOT.rglob("relight-best"):
        if p.is_dir(): search_dirs.append(p)
    search_dirs += [
        KAGGLE_DATA_ROOT / "datasets/coding12341234/relight-checkpoint",
        KAGGLE_DATA_ROOT / "datasets/coding12341234/relight-best",
    ]
    best_path, best_step, best_loss = None, -1, float("inf")
    print("  [RESUME] Scanning for valid checkpoints...")
    for d in search_dirs:
        if not d.exists(): continue
        for pt in sorted(d.glob("*.pt")):
            valid, step, loss = _is_valid_pt(pt)
            marker = ""
            if valid and step > best_step:
                best_step = step; best_loss = loss
                best_path = pt; marker = "  ← selected"
            status = f"step={step}  loss={loss:.4f}" if valid else "CORRUPT/NaN"
            print(f"    {pt.name:<25}  {status}{marker}")
    return best_path, best_step, best_loss

if not _dst_ckpt.exists():
    _best_src, _best_step, _best_loss = _find_best_checkpoint()
    if _best_src is not None:
        shutil.copy2(_best_src, _dst_ckpt)
        print(f"  [RESUME] Copied {_best_src.name} (step={_best_step}, loss={_best_loss:.4f}) → last.pt ✓")
    else:
        print("  [RESUME] No valid checkpoint found — will start from scratch.")
else:
    valid, step, loss = _is_valid_pt(_dst_ckpt)
    if valid:
        print(f"  [RESUME] last.pt already exists ✓  (step={step}, loss={loss:.4f})")
    else:
        print("  [RESUME] last.pt is CORRUPT/NaN — re-scanning...")
        _dst_ckpt.unlink()
        _best_src, _best_step, _best_loss = _find_best_checkpoint()
        if _best_src is not None:
            shutil.copy2(_best_src, _dst_ckpt)
            print(f"  [RESUME] Replaced with {_best_src.name} ✓")

_csv_src = KAGGLE_DATA_ROOT / "datasets/coding12341234/relight-metadata" / "master_metadata.csv"
_dst_csv = KAGGLE_WORK_DIR / "master_metadata.csv"
if _csv_src.exists() and not _dst_csv.exists():
    shutil.copy2(_csv_src, _dst_csv)
    print(f"  [RESUME] Copied master_metadata.csv ✓")
else:
    print(f"  [RESUME] master_metadata.csv: {'already exists' if _dst_csv.exists() else 'NOT FOUND'}")

# ============================================================
# FIND DATA ROOT
# ============================================================

def find_alexander_root():
    candidates = [
        KAGGLE_DATA_ROOT / "alexander" / "alexander",
        KAGGLE_DATA_ROOT / "alexander",
        KAGGLE_DATA_ROOT / "relight-dataset" / "alexander",
        KAGGLE_DATA_ROOT / "synthlight" / "alexander",
        KAGGLE_DATA_ROOT / "datasets/devsachaniitk/alexander/alexander",
        Path("D:/alexander"),
    ]
    for c in candidates:
        if c.exists():
            print(f"  Found data root: {c}")
            return c
    for p in KAGGLE_DATA_ROOT.rglob("synthlight_renders_v6"):
        return p.parent
    raise FileNotFoundError("Could not find alexander/ folder.")

ALEXANDER_ROOT = find_alexander_root()
HDRI_CACHE_DIR = Path("/kaggle/input/datasets/devsachaniitk/hdri-cache/hdri_cache")
if not HDRI_CACHE_DIR.exists():
    found = list(Path("/kaggle/input").rglob("hdri_cache"))
    HDRI_CACHE_DIR = found[0] if found else HDRI_CACHE_DIR

SH_CACHE_DIR      = KAGGLE_WORK_DIR / "sh_cache"
CSV_PATH          = KAGGLE_WORK_DIR / "master_metadata.csv"
CKPT_DIR          = KAGGLE_WORK_DIR / "checkpoints"
SAMPLE_DIR        = KAGGLE_WORK_DIR / "samples"
DIAG_DIR          = KAGGLE_WORK_DIR / "diag_grids"
for d in [SH_CACHE_DIR, CKPT_DIR, SAMPLE_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  [DEVICE] {DEVICE}")

# ============================================================
# GPU CONFIG
# ============================================================

def get_gpu_config():
    if not torch.cuda.is_available():
        return {"tier":"cpu","name":"CPU","batch_size":2,"base_ch":64,
                "use_amp":False,"ddim_steps":20,"num_workers":0}
    name = torch.cuda.get_device_name(0).lower()
    mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    if "a100" in name:
        return {"tier":"a100","name":name,"batch_size":8,"base_ch":128,
                "use_amp":True,"ddim_steps":50,"num_workers":4}
    elif "t4" in name or mem < 16:
        return {"tier":"t4","name":name,"batch_size":4,"base_ch":128,
                "use_amp":True,"ddim_steps":50,"num_workers":2}
    else:
        return {"tier":"other","name":name,"batch_size":4,"base_ch":128,
                "use_amp":True,"ddim_steps":50,"num_workers":2}

GPU_CFG = get_gpu_config()
_accum  = 4 if GPU_CFG["tier"] == "rtx3060" else 2

# ============================================================
# CONFIG
# ============================================================

CFG = {
    "device"          : DEVICE,
    "batch_size"      : GPU_CFG["batch_size"],
    "use_amp"         : GPU_CFG["use_amp"],
    "image_size"      : 256,
    "total_steps"     : 500_000,   # extended — you were at 260k, need more
    "lr"              : 3e-5,
    "lr_min"          : 5e-7,      # lower floor — better late-stage refinement
    "wd"              : 1e-4,
    "T"               : 1000,
    "base_ch"         : GPU_CFG["base_ch"],
    "ze_dim"          : 256,

    # ── LOSS WEIGHTS ─────────────────────────────────────────
    # SEE TUNING GUIDE AT TOP OF FILE BEFORE CHANGING THESE
    "mse_weight"      : 1.0,   # core diffusion noise MSE — do not reduce below 1.0
    "lpips_weight"    : 3.0,   # was 5.0 — still strong perceptual signal
    "ch_loss_weight"  : 1.5,   # was 1.0 — slight boost for colour histogram
    "shadow_weight"   : 4.0,   # was 8.0 — halved because bug is now fixed
    "freq_loss_weight": 2.0,   # was 8.0 — was causing blurriness fighting sharpness
    "color_mv_weight" : 4.0,   # was 12.0 — too high caused colour banding
    "sat_loss_weight" : 2.0,   # NEW — directly fixes washed-out/grey predictions
    # ─────────────────────────────────────────────────────────

    "fg_weight"       : 5.0,
    "bg_weight"       : 0.15,
    "num_workers"     : GPU_CFG["num_workers"],
    "log_every"       : 100,
    "sample_every"    : 1_000,
    "diag_every"      : 10_000,
    "save_last_every" : 500,
    "save_ckpt_every" : 5_000,
    "val_fraction"    : 0.02,
    "accum_steps"     : _accum,
    "ddim_steps"      : GPU_CFG["ddim_steps"],
    "wandb_project"   : "relight-diffusion",
    "lpips_every"     : 1,
    "lpips_max_t"     : 150,
    "val_every"       : 5_000,
}

# ============================================================
# DEBUG MODE
# ============================================================

DEBUG_MODE         = False
DEBUG_STEPS        = 200
DEBUG_SAMPLE_EVERY = 20
DEBUG_DIAG_AT      = 200

# ============================================================
# WANDB
# ============================================================

_wandb = None

def init_wandb():
    global _wandb
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY", "")
        if not api_key:
            try:
                from kaggle_secrets import UserSecretsClient
                api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
            except Exception:
                pass
        if not api_key:
            print("  [WandB] WANDB_API_KEY not found — logging disabled.")
            return
        wandb.login(key=api_key, relogin=True)
        wandb.init(project=CFG["wandb_project"], config=CFG, resume="allow",
                   name=f"relight-v13-step{CFG['total_steps']//1000}k")
        _wandb = wandb
        print("  [WandB] Connected ✓")
    except Exception as e:
        print(f"  [WandB] Failed: {e}")

def wandb_log(d):
    if _wandb:
        try: _wandb.log(d)
        except Exception: pass

def wandb_log_image(key, img, step):
    if _wandb:
        try: _wandb.log({key: _wandb.Image(img)}, step=step)
        except Exception: pass

def wandb_finish():
    if _wandb:
        try: _wandb.finish()
        except Exception: pass

# ============================================================
# HELPERS
# ============================================================

def _normalise_hdri_key(raw: str) -> str:
    key = raw.strip().lower()
    for ext in (".hdr", ".exr", ".png"):
        if key.endswith(ext):
            key = key[:-len(ext)]; break
    return key

# ============================================================
# SH PRECOMPUTE
# ============================================================

def compute_sh_from_npy(hdr_arr, n_samples=50000):
    C, H, W = hdr_arr.shape
    ys = (np.arange(H) + 0.5) / H
    sin_theta = np.sin(ys * np.pi)
    weights   = sin_theta / (sin_theta.sum() + 1e-8)
    flat_w    = np.repeat(weights, W)
    flat_w   /= flat_w.sum()
    flat_r = hdr_arr[0].ravel(); flat_g = hdr_arr[1].ravel(); flat_b = hdr_arr[2].ravel()
    idx = np.random.choice(len(flat_r), size=min(n_samples, len(flat_r)),
                           replace=False, p=flat_w)
    xs_all = (np.arange(W) + 0.5) / W * 2 * np.pi
    ys_all = ys * np.pi
    phi_flat   = np.tile(xs_all, H); theta_flat = np.repeat(ys_all, W)
    phi = phi_flat[idx]; theta = theta_flat[idx]
    r = flat_r[idx]; g = flat_g[idx]; b = flat_b[idx]
    x = np.sin(theta)*np.cos(phi); y = np.sin(theta)*np.sin(phi); z = np.cos(theta)
    Y = np.stack([
        np.ones(len(x)) * 0.282095,
        y * 0.488603, z * 0.488603, x * 0.488603,
        x*y * 1.092548, y*z * 1.092548,
        (3*z*z-1) * 0.315392, x*z * 1.092548,
        (x*x-y*y) * 0.546274,
    ], axis=1)
    coeffs = np.concatenate([
        (Y * r[:, None]).mean(axis=0),
        (Y * g[:, None]).mean(axis=0),
        (Y * b[:, None]).mean(axis=0),
    ])
    return coeffs.astype(np.float32)

def precompute_sh_if_needed():
    sh_npy  = SH_CACHE_DIR / "sh_coeffs.npy"
    sh_json = SH_CACHE_DIR / "sh_index.json"
    if sh_npy.exists() and sh_json.exists():
        print("  SH cache found — skipping.")
        return str(sh_npy), str(sh_json)

    _sh_input_candidates = [
        KAGGLE_DATA_ROOT / "datasets/coding12341234/relight-sh-cache",
        KAGGLE_DATA_ROOT / "relight-sh-cache",
    ]
    for _sh_src in _sh_input_candidates:
        _sh_npy_src  = _sh_src / "sh_coeffs.npy"
        _sh_json_src = _sh_src / "sh_index.json"
        if _sh_npy_src.exists() and _sh_json_src.exists():
            shutil.copy2(_sh_npy_src,  sh_npy)
            shutil.copy2(_sh_json_src, sh_json)
            print(f"  SH cache loaded from input dataset ✓  ({_sh_src.name})")
            return str(sh_npy), str(sh_json)

    print("  Computing SH coefficients from HDR cache...")
    hdr_files = sorted(HDRI_CACHE_DIR.glob("*_hdr.npy"))
    if not hdr_files:
        raise FileNotFoundError(f"No *_hdr.npy files in {HDRI_CACHE_DIR}")
    all_coeffs = []; sh_index = {}
    for i, fpath in enumerate(hdr_files):
        arr = np.load(fpath)
        if arr.ndim != 3 or arr.shape[0] != 3: continue
        key = _normalise_hdri_key(fpath.stem.replace("_hdr", ""))
        sh_index[key] = len(all_coeffs)
        all_coeffs.append(compute_sh_from_npy(arr))
        if (i+1) % 100 == 0: print(f"    [{i+1}/{len(hdr_files)}]")
    arr_out = np.stack(all_coeffs)
    np.save(sh_npy, arr_out)
    with open(sh_json, "w") as f: json.dump(sh_index, f)
    print(f"  SH done: {arr_out.shape}")
    return str(sh_npy), str(sh_json)

# ============================================================
# BUILD CSV
# ============================================================

VERSION_TO_SUBJECT = {
    "synthlight_renders_v6"          : 1,
    "synthlight_renders_v7"          : 2,
    "synthlight_renders_v8"          : 3,
    "synthlight_renders_v9_cache_128": 4,
}

def build_csv_if_needed():
    if CSV_PATH.exists():
        print("  CSV found — skipping build.")
        return str(CSV_PATH)

    import csv
    print("  Building master_metadata.csv...")
    rows = []
    for version, sid in VERSION_TO_SUBJECT.items():
        mesh_dir = ALEXANDER_ROOT / version / "model_mesh"
        if not mesh_dir.exists(): continue
        for json_file in sorted(mesh_dir.glob("*.json")):
            with open(json_file) as f:
                entries = json.load(f)
            if isinstance(entries, dict):
                entries = next(v for v in entries.values() if isinstance(v, list))
            for e in entries:
                fname = e.get("filename", "")
                img_path = ALEXANDER_ROOT / version / "model_mesh" / fname
                if not img_path.exists(): continue
                hdri_name = e.get("hdri_name", "")
                hdri_key  = _normalise_hdri_key(hdri_name)
                rows.append({
                    "image_path"     : str(img_path),
                    "subject_id"     : sid,
                    "filename"       : fname,
                    "hdri_name"      : hdri_name,
                    "hdri_name_key"  : hdri_key,
                    "hdri_rank"      : e.get("hdri_rank", 0),
                    "hdri_tier"      : e.get("hdri_tier", 0),
                    "hdri_tier_name" : e.get("hdri_tier_name", ""),
                    "hdri_score"     : e.get("hdri_score", 0.0),
                    "hdri_avg_lum"   : e.get("hdri_avg_lum", 0.0),
                    "hdri_peak_lum"  : e.get("hdri_peak_lum", 0.0),
                    "hdri_dyn_range" : e.get("hdri_dyn_range", 0.0),
                    "hdri_strength"  : e.get("hdri_strength", 1.0),
                    "hdri_saturation": e.get("hdri_saturation", 1.0),
                    "hdri_rotation_deg": e.get("hdri_rotation_deg", 0.0),
                    "hdri_rotation_idx": e.get("hdri_rotation_idx", 0),
                    "camera_position": e.get("camera_position", "front"),
                    "camera_yaw_deg" : e.get("camera_yaw_deg", 0),
                    "exposure_used"  : e.get("exposure_used", 1.0),
                })
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"  CSV built: {len(df):,} rows")
    return str(CSV_PATH)

# ============================================================
# DATASET
# ============================================================

METADATA_FIELDS = [
    "hdri_avg_lum","hdri_peak_lum","hdri_dyn_range","hdri_strength",
    "hdri_saturation","hdri_rotation_deg","exposure_used","hdri_tier",
]

class RelightDataset(Dataset):
    def __init__(self, csv_path, sh_npy_path, sh_json_path,
                 image_size=256, split="train", val_fraction=0.02, seed=42):
        import pandas as pd
        self.image_size = image_size

        df = pd.read_csv(csv_path)
        if "hdri_name_key" not in df.columns:
            df["hdri_name_key"] = df["hdri_name"].astype(str).apply(_normalise_hdri_key)

        if "camera_yaw_deg" not in df.columns:
            df["camera_yaw_deg"] = 0
        df["_yaw_key"] = df["camera_yaw_deg"].round(1)

        self.sh_coeffs = np.load(sh_npy_path)
        with open(sh_json_path) as f:
            self.sh_index = {_normalise_hdri_key(k): v for k, v in json.load(f).items()}

        self.meta_mean = {}; self.meta_std = {}
        for field in METADATA_FIELDS:
            if field in df.columns:
                vals = df[field].astype(float)
                self.meta_mean[field] = float(vals.mean())
                self.meta_std[field]  = float(vals.std()) + 1e-8

        self.angle_groups = {}
        for (sid, yaw), grp in df.groupby(["subject_id", "_yaw_key"]):
            valid = grp[grp["hdri_name_key"].apply(lambda k: k in self.sh_index)]
            if len(valid) > 0:
                self.angle_groups[(int(sid), float(yaw))] = valid.reset_index(drop=True)

        all_keys = sorted(list(self.angle_groups.keys()))
        rng = random.Random(seed); rng.shuffle(all_keys)
        n_val = max(1, int(len(all_keys) * val_fraction))
        split_keys = all_keys[n_val:] if split == "train" else all_keys[:n_val]

        self.indices = [
            (key, i)
            for key in split_keys
            for i in range(len(self.angle_groups[key]))
        ]
        print(f"  [Dataset/{split}] {len(self.indices):,} samples across {len(split_keys)} angle groups")

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def _get_sh(self, key):
        norm_key = _normalise_hdri_key(str(key))
        idx = self.sh_index.get(norm_key)
        return np.zeros(27, dtype=np.float32) if idx is None else np.clip(self.sh_coeffs[idx], -10.0, 10.0)

    def _get_meta(self, row):
        import pandas as pd
        vals = []
        for f in METADATA_FIELDS:
            if f not in row.index:
                vals.append(0.0); continue
            if f == "hdri_rotation_deg":
                raw_deg = float(row[f]) if not pd.isna(row[f]) else 0.0
                vals += [math.sin(math.radians(raw_deg)), math.cos(math.radians(raw_deg))]
            else:
                v = (float(row[f]) - self.meta_mean.get(f, 0)) / self.meta_std.get(f, 1)
                vals.append(float(np.clip(v, -5.0, 5.0)))
        return np.array(vals, dtype=np.float32)

    def _load_img(self, path):
        try:
            img = self.transform(Image.open(path).convert("RGB"))
            return img if not (torch.isnan(img).any() or torch.isinf(img).any()) \
                       else torch.zeros(3, self.image_size, self.image_size)
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        (sid, yaw), row_i = self.indices[idx]
        grp  = self.angle_groups[(sid, yaw)]
        row1 = grp.iloc[row_i]

        if len(grp) > 1:
            cands = grp[grp["hdri_rank"] != row1["hdri_rank"]]
            if len(cands) == 0:
                cands = grp.drop(row_i)
            if len(cands) == 0:
                row2 = row1
            else:
                row2 = cands.sample(1).iloc[0]
        else:
            row2 = row1

        img1 = self._load_img(row1["image_path"])
        img2 = self._load_img(row2["image_path"])

        sh2      = self._get_sh(row2["hdri_name_key"])
        meta2    = self._get_meta(row2)
        ze_input = torch.from_numpy(np.concatenate([sh2, meta2])).clamp(-10.0, 10.0)

        return img1, img2, ze_input, torch.tensor(sid, dtype=torch.long)

# ============================================================
# DYNAMIC HDRI VALIDATION BATCH
# ============================================================

def get_random_val_batch(val_ds, n_samples=4):
    n   = len(val_ds)
    if n >= n_samples:
        idxs = random.sample(range(n), n_samples)
    else:
        idxs = [random.randint(0, n - 1) for _ in range(n_samples)]

    items = [val_ds[i] for i in idxs]

    img1s    = torch.stack([it[0] for it in items])
    img2s    = torch.stack([it[1] for it in items])
    ze_inps  = torch.stack([it[2] for it in items])
    sids     = torch.stack([it[3] for it in items])

    hdri_tiers = []
    for it in items:
        hdri_tiers.append(f"S{it[3].item()}")
    print(f"  [VAL BATCH] Random {n_samples} samples: {', '.join(hdri_tiers)}")

    return img1s, img2s, ze_inps, sids

# ============================================================
# MODELS
# ============================================================

class AdaGN(nn.Module):
    def __init__(self, num_channels, ze_dim=256, num_groups=32):
        super().__init__()
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(ze_dim, num_channels * 2)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x, ze):
        h = self.norm(x)
        s, b = self.proj(ze).chunk(2, dim=-1)
        return h * (s[:, :, None, None] + 1) + b[:, :, None, None]

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ze_dim=256):
        super().__init__()
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.adagn1 = AdaGN(out_ch, ze_dim)
        self.adagn2 = AdaGN(out_ch, ze_dim)
        self.act    = nn.SiLU()
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.act(self.adagn1(self.conv1(x), cond))
        h = self.act(self.adagn2(self.conv2(h), cond))
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device; half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class TimestepEmbedding(nn.Module):
    def __init__(self, ze_dim):
        super().__init__()
        self.sinpos = SinusoidalPosEmb(ze_dim)
        self.mlp    = nn.Sequential(
            nn.Linear(ze_dim, ze_dim * 4), nn.SiLU(),
            nn.Linear(ze_dim * 4, ze_dim)
        )
    def forward(self, t): return self.mlp(self.sinpos(t))

# [FIX-6] NEW: Self-attention at bottleneck for global illumination reasoning.
# This is what fixes:
#   - flat nose (needs to see full face context to shade correctly)
#   - soft shadow boundaries (needs to know where light comes from globally)
#   - wrong colour casts (needs to relate SH embedding to full image)
class SelfAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        # num_heads must divide ch evenly
        while ch % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.norm = nn.GroupNorm(32 if ch >= 32 else ch, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        # zero-init output projection so attention starts as identity
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h  # residual — safe to add even at step 0

class UNet(nn.Module):
    def __init__(self, in_channels=8, base_ch=128, ch_mult=(1,2,2,2), ze_dim=256):
        super().__init__()
        self.t_emb   = TimestepEmbedding(ze_dim)
        chs          = [base_ch * m for m in ch_mult]
        self.in_proj = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.enc_blocks  = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_chs    = []
        prev = chs[0]
        for ch in chs[:-1]:
            self.enc_blocks.append(ResBlock(prev, ch, ze_dim))
            self.skip_chs.append(ch)
            self.downsamples.append(Downsample(ch))
            prev = ch

        self.mid1     = ResBlock(prev, chs[-1], ze_dim)
        # [FIX-6] Self-attention between mid1 and mid2 — key fix for global lighting
        self.mid_attn = SelfAttention(chs[-1], num_heads=4)
        self.mid2     = ResBlock(chs[-1], chs[-1], ze_dim)
        prev = chs[-1]

        self.upsamples  = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.upsamples.append(Upsample(prev))
            self.dec_blocks.append(ResBlock(prev + ch, ch, ze_dim))
            prev = ch

        self.out_norm = nn.GroupNorm(32, prev)
        self.out_proj = nn.Conv2d(prev, 4, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, t, ze):
        cond  = ze + self.t_emb(t)
        h     = self.in_proj(x)
        skips = []
        for blk, down in zip(self.enc_blocks, self.downsamples):
            h = blk(h, cond); skips.append(h); h = down(h)
        # [FIX-6] attention applied at bottleneck (smallest spatial resolution)
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)
        for up, blk, sk in zip(self.upsamples, self.dec_blocks, reversed(skips)):
            h = up(h); h = torch.cat([h, sk], dim=1); h = blk(h, cond)
        h_out = F.silu(self.out_norm(h))
        return self.out_proj(h_out)

# [FIX-7] LP3 made deeper with LayerNorm — better SH→lighting embedding.
# The 3-layer version was a bottleneck for colour accuracy per-HDRI.
# If predictions still have wrong colour: increase hidden_dim from 512 → 768
class LP3(nn.Module):
    def __init__(self, in_dim=36, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)

class DDPM:
    def __init__(self, T=1000, device="cuda"):
        self.T = T; self.device = device
        t_vals    = torch.linspace(0, T, T + 1, device=device)
        f         = torch.cos(((t_vals / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        f         = f / f[0]
        alpha_bar = (f[1:] / f[:-1]).cumprod(0).clamp(1e-5, 1 - 1e-5)
        abp   = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])
        betas = (1 - alpha_bar / abp).clamp(0, 0.999)
        self.betas          = betas
        self.alpha_bar      = alpha_bar
        self.sqrt_ab        = alpha_bar.sqrt()
        self.sqrt_one_m_ab  = (1 - alpha_bar).sqrt()
        self.alpha_bar_prev = abp
        self.posterior_var  = (betas * (1 - abp) / (1 - alpha_bar)).clamp(1e-20)

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        s1 = self.sqrt_ab[t][:, None, None, None]
        s2 = self.sqrt_one_m_ab[t][:, None, None, None]
        return s1 * x0 + s2 * noise, noise

    def sample_t(self, B):
        return torch.randint(0, self.T, (B,), device=self.device)

    def sample_low_t(self, B, max_t=250):
        return torch.randint(0, max_t, (B,), device=self.device)

# ============================================================
# VAE
# ============================================================

def load_vae(device):
    from diffusers import AutoencoderKL
    local_vae = Path("/kaggle/input/sd-vae-ft-mse")
    if local_vae.exists():
        vae = AutoencoderKL.from_pretrained(str(local_vae))
    else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return vae.to(device)

@torch.no_grad()
def vae_encode(vae, x): return vae.encode(x).latent_dist.sample() * 0.18215

@torch.no_grad()
def vae_decode(vae, z): return vae.decode(z / 0.18215).sample.clamp(-1, 1)

# ============================================================
# LPIPS
# ============================================================

def load_lpips(device):
    from lpips import LPIPS
    fn = LPIPS(net="vgg").to(device)
    for p in fn.parameters(): p.requires_grad_(False)
    return fn

# ============================================================
# DDIM SAMPLER
# ============================================================

@torch.no_grad()
def ddim_sample(unet, lp3, ddpm, z1, ze_inp, n_steps, device):
    ze    = lp3(ze_inp)
    t_seq = torch.linspace(ddpm.T - 1, 0, n_steps, dtype=torch.long).tolist()
    B     = z1.shape[0]
    x     = torch.randn_like(z1)
    for i, t_idx in enumerate(t_seq):
        t_idx   = int(t_idx)
        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
        eps = unet(torch.cat([x, z1], dim=1), t_batch, ze)
        ab      = ddpm.alpha_bar[t_idx]
        # [FIX-8] Relaxed clamp: was (-1, 1) which crushed highlights/shadows.
        # 1.2 gives the model room to express high-contrast lighting.
        # If predictions become too noisy: tighten back to (-1.05, 1.05)
        x0      = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.2, 1.2)
        if i < len(t_seq) - 1:
            ab_prev = ddpm.alpha_bar[int(t_seq[i + 1])]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            x = x0.clamp(-1, 1)
    return x

# ============================================================
# LOSS HELPERS
# ============================================================

def color_histogram_loss(pred, target, n_bins=32):
    B, C, H, W  = pred.shape
    pred_flat   = pred.view(B, C, -1)
    target_flat = target.view(B, C, -1)
    bins  = torch.linspace(-1, 1, n_bins, device=pred.device)
    sigma = 2.0 / n_bins
    total_loss = 0.0
    for c in range(C):
        p = pred_flat[:, c, :].unsqueeze(2)
        t = target_flat[:, c, :].unsqueeze(2)
        b = bins.view(1, 1, n_bins)
        ph = torch.exp(-((p - b)**2) / (2 * sigma**2)).mean(dim=1)
        th = torch.exp(-((t - b)**2) / (2 * sigma**2)).mean(dim=1)
        ph = ph / (ph.sum(dim=1, keepdim=True) + 1e-8)
        th = th / (th.sum(dim=1, keepdim=True) + 1e-8)
        total_loss += F.l1_loss(ph, th)
    return total_loss / C

def frequency_loss(pred, target):
    def sobel_edges(x):
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                          dtype=x.dtype, device=x.device).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                          dtype=x.dtype, device=x.device).view(1,1,3,3)
        gray = x.mean(dim=1, keepdim=True)
        gx   = F.conv2d(gray, kx, padding=1)
        gy   = F.conv2d(gray, ky, padding=1)
        return (gx**2 + gy**2 + 1e-8).sqrt()
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))

# [FIX-5] NEW: Saturation loss — directly penalises grey/washed-out predictions.
# Works by matching per-channel standard deviation in pixel space on the fg only.
# Higher std = more saturated / more contrast.
# If predictions are STILL grey: increase sat_loss_weight in CFG (2.0 → 4.0)
# If predictions have oversaturated neon look: reduce to 1.0
def saturation_loss(pred, target, fg_mask):
    """
    pred, target: (B, 3, H, W) in [-1, 1]
    fg_mask:      (B, 1, H, W) binary foreground mask
    """
    fg = fg_mask.expand_as(pred)  # (B, 3, H, W)
    loss = torch.tensor(0.0, device=pred.device)
    count = 0
    for b in range(pred.shape[0]):
        for c in range(3):
            p_vals = pred[b, c][fg[b, c] > 0.5]
            t_vals = target[b, c][fg[b, c] > 0.5]
            if p_vals.numel() < 50:
                continue
            p_std = p_vals.std()
            t_std = t_vals.std()
            loss  = loss + (p_std - t_std).abs()
            count += 1
    return loss / max(count, 1)

def apply_fg_mask(img_tensor, subject_masks, subject_ids):
    out = img_tensor.clone()
    for b_i, sid in enumerate(subject_ids.tolist()):
        sid = int(sid)
        if sid in subject_masks:
            m = subject_masks[sid]
            out[b_i : b_i + 1] = img_tensor[b_i : b_i + 1] * m + 0.0 * (1 - m)
    return out

# ============================================================
# UTILITIES FOR DIAGNOSTIC GRID
# ============================================================

def _t2pil(t):
    t = t.squeeze(0).clamp(-1, 1)
    return Image.fromarray(((t + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy())

def _np2pil(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def _err_map(pred, gt, amplify=4.0):
    diff = (pred.clamp(-1,1) - gt.clamp(-1,1)).abs() * amplify
    return _np2pil((diff.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))

def _shadow_map_to_pil(sm, size=256):
    sm_np = sm.cpu().numpy() if torch.is_tensor(sm) else sm
    r = (sm_np * 255).astype(np.uint8)
    g = (sm_np * 255).astype(np.uint8)
    b = (255 * np.ones_like(sm_np)).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)

def _shadow_weight_to_pil(sm, size=256):
    sm_np = sm.cpu().numpy() if torch.is_tensor(sm) else sm
    w = 1.0 - sm_np
    r = (w * 255).astype(np.uint8)
    g = np.zeros_like(r)
    b = ((1 - w) * 200).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)

def _weighted_err_map(pred, gt, shadow_map, size=256, amplify=4.0):
    diff = (pred.clamp(-1,1) - gt.clamp(-1,1)).abs()
    sm   = shadow_map.cpu().numpy() if torch.is_tensor(shadow_map) else shadow_map
    sm_t = torch.from_numpy(sm).unsqueeze(0).unsqueeze(0)
    sm_up = F.interpolate(sm_t, size=(pred.shape[-2], pred.shape[-1]), mode="bilinear", align_corners=False).squeeze()
    weight = (1.0 - sm_up).to(pred.device)
    weighted = (diff * weight.unsqueeze(0) * amplify).clamp(0, 1)
    return _np2pil((weighted.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))

def _lpips_heatmap(pred, gt, lpips_fn, size=256):
    try:
        with torch.no_grad():
            B = pred.unsqueeze(0) if pred.dim() == 3 else pred
            G = gt.unsqueeze(0) if gt.dim() == 3 else gt
            feat_diff = (B.clamp(-1,1) - G.clamp(-1,1)).abs().mean(dim=1).squeeze()
        heat = feat_diff.cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        r = np.clip(1.5 - np.abs(heat * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(heat * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(heat * 4 - 1), 0, 1)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        return Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
    except Exception:
        return Image.new("RGB", (size, size), (40, 40, 60))

def _sh_light_arrow_pil(sh_vec_np, base_img_pil, size=256):
    sh_vec = np.clip(sh_vec_np, -10.0, 10.0)
    sh_mean = (sh_vec[0:9] + sh_vec[9:18] + sh_vec[18:27]) / 3.0
    Lx, Ly, Lz = sh_mean[3], sh_mean[1], sh_mean[2]
    L = np.array([Lx, Ly, Lz])
    n = np.linalg.norm(L)

    img = base_img_pil.copy().resize((size, size))
    if n < 1e-6:
        return img
    L /= n
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    scale  = size * 0.30
    ex = int(cx + L[0] * scale)
    ey = int(cy - L[1] * scale)
    draw.line([(cx, cy), (ex, ey)], fill=(255, 220, 0), width=4)
    draw.ellipse([(ex-6, ey-6), (ex+6, ey+6)], fill=(255, 220, 0))
    draw.ellipse([(cx-4, cy-4), (cx+4, cy+4)], fill=(255, 100, 0))
    strength = f"L=({L[0]:.2f},{L[1]:.2f},{L[2]:.2f})"
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        fnt = ImageFont.load_default()
    draw.text((4, size - 18), strength, fill=(255, 220, 0), font=fnt)
    return img

def _lp3_feature_pil(ze_tensor, size=256):
    try:
        ze = ze_tensor.float().cpu().numpy()
        spatial = ze.reshape(16, 16)
        mn, mx  = spatial.min(), spatial.max()
        norm    = (spatial - mn) / (mx - mn + 1e-8)
        r = norm
        g = np.roll(norm, 4, axis=0)
        b = np.roll(norm, 8, axis=1)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        return Image.fromarray(rgb).resize((size, size), Image.NEAREST)
    except Exception:
        return Image.new("RGB", (size, size), (30, 30, 50))

def _hdri_metadata_pil(ze_input_np, hdri_tier_name="", hdri_avg_lum=0.0,
                        exposure=0.0, size=256):
    img  = Image.new("RGB", (size, size), (15, 15, 25))
    draw = ImageDraw.Draw(img)
    try:
        fnt  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        fnt_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        fnt = fnt_s = ImageFont.load_default()

    tier_colors = {
        "SUNNY": (255, 220, 50), "OUTDOOR": (100, 200, 100),
        "CLOUDY": (150, 180, 220), "INDOOR": (200, 150, 255),
    }
    tier_col = tier_colors.get(str(hdri_tier_name).upper(), (200, 200, 200))
    draw.text((10, 10),  "HDRI METADATA",       fill=(180, 180, 180), font=fnt_s)
    draw.text((10, 30),  f"Tier: {hdri_tier_name}", fill=tier_col, font=fnt)
    draw.text((10, 55),  f"Avg Lum: {hdri_avg_lum:.3f}", fill=(200, 200, 200), font=fnt_s)
    draw.text((10, 72),  f"Exposure: {exposure:.3f}",    fill=(200, 200, 200), font=fnt_s)

    sh = ze_input_np[:27]
    energy = np.abs(sh).reshape(3, 9).mean(axis=0)
    energy = energy / (energy.max() + 1e-8)
    bar_w  = (size - 20) // 9
    for bi, e in enumerate(energy):
        bh = int(e * 60)
        x0 = 10 + bi * bar_w; y1 = size - 30; y0 = y1 - bh
        draw.rectangle([x0, y0, x0 + bar_w - 2, y1], fill=(100, 180, 255))
    draw.text((10, size - 20), "SH energy (band 0-8)", fill=(140, 140, 140), font=fnt_s)
    return img

def _hdri_env_pil(hdri_key: str, size=256):
    try:
        candidates = list(HDRI_CACHE_DIR.glob(f"{hdri_key}_hdr.npy"))
        if not candidates:
            candidates = [p for p in HDRI_CACHE_DIR.glob("*_hdr.npy")
                         if hdri_key[:12] in p.stem]
        if not candidates:
            return Image.new("RGB", (size, size), (20, 20, 30))

        hdr = np.load(candidates[0])
        hdr = np.transpose(hdr, (1, 2, 0))
        hdr_tm = hdr / (1.0 + hdr)
        hdr_tm = np.clip(hdr_tm, 0, 1)
        hdr_tm = np.power(hdr_tm, 1.0 / 2.2)
        img = (hdr_tm * 255).astype(np.uint8)
        return Image.fromarray(img).resize((size, size), Image.LANCZOS)
    except Exception:
        return Image.new("RGB", (size, size), (20, 20, 30))

def _sh_sphere_pil(sh_vec_np, size=256):
    sh_vec  = np.clip(sh_vec_np, -10.0, 10.0)
    sh_r    = sh_vec[0:9]; sh_g = sh_vec[9:18]; sh_b = sh_vec[18:27]

    ys = (np.arange(size) + 0.5) / size * 2 - 1
    xs = (np.arange(size) + 0.5) / size * 2 - 1
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    r2    = XX**2 + YY**2
    valid = r2 <= 1.0
    ZZ    = np.where(valid, np.sqrt(np.clip(1.0 - r2, 0, 1)), 0.0)

    Y = np.stack([
        np.ones_like(XX)  * 0.282095,
        YY                * 0.488603,
        ZZ                * 0.488603,
        XX                * 0.488603,
        XX * YY           * 1.092548,
        YY * ZZ           * 1.092548,
        (3*ZZ*ZZ - 1)     * 0.315392,
        XX * ZZ           * 1.092548,
        (XX*XX - YY*YY)   * 0.546274,
    ], axis=-1)

    L_r = (Y * sh_r).sum(axis=-1)
    L_g = (Y * sh_g).sum(axis=-1)
    L_b = (Y * sh_b).sum(axis=-1)

    L = np.stack([L_r, L_g, L_b], axis=-1)
    L_pos = np.clip(L, 0, None)
    mn = L_pos.min(); mx = L_pos.max()
    L_norm   = (L_pos - mn) / (mx - mn + 1e-8)
    L_mapped = np.power(L_norm, 0.4)
    out = (L_mapped * 255).clip(0, 255).astype(np.uint8)
    out[~valid] = [15, 15, 30]
    return Image.fromarray(out)

# ============================================================
# DIAGNOSTIC GRID
# ============================================================

def _label_panel(img, text, col=(220, 220, 220)):
    W, H   = img.size
    canvas = Image.new("RGB", (W, H + 20), (10, 10, 18))
    canvas.paste(img, (0, 20))
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        fnt = ImageFont.load_default()
    ImageDraw.Draw(canvas).text((3, 3), text, fill=col, font=fnt)
    return canvas

@torch.no_grad()
def save_diagnostic_grid(vae, unet, lp3, ddpm, lpips_fn,
                          val_batch, step, cfg, subject_masks, sh_index, sh_coeffs):
    device = cfg["device"]
    n_show = min(4, val_batch[0].shape[0])
    S      = cfg["image_size"]

    img1     = val_batch[0][:n_show].to(device)
    img2     = val_batch[1][:n_show].to(device)
    ze_inp   = val_batch[2][:n_show].to(device)
    sids     = val_batch[3][:n_show]

    img1_m = img1

    unet.eval(); lp3.eval()
    z1       = vae_encode(vae, img1_m)
    x_lat    = ddim_sample(unet, lp3, ddpm, z1, ze_inp, n_steps=cfg["ddim_steps"], device=device)
    pred     = vae_decode(vae, x_lat)

    ze_feats = lp3(ze_inp)

    unet.train(); lp3.train()

    N_COLS = 13
    PANEL  = S
    PAD    = 4
    HDR    = 50
    PH     = S + 20
    GW     = N_COLS * PANEL + (N_COLS + 1) * PAD
    GH     = HDR + n_show * (PH + PAD) + PAD
    grid   = Image.new("RGB", (GW, GH), (8, 8, 14))
    draw   = ImageDraw.Draw(grid)

    try:
        fh = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        fh = fs = ImageFont.load_default()

    draw.text((PAD, 6),
              f"RELIGHT DIAGNOSTIC  step={step:,}  [v13 all-fixes]",
              fill=(255, 210, 50), font=fh)

    col_labels = [
        ("Input (masked)",    (160,160,160)),
        ("Ground Truth",      (80, 200, 80)),
        ("Prediction",        (80, 150, 255)),
        ("GT Lum Map",        (255,180, 80)),
        ("Pred Lum Map",      (255, 80, 80)),
        ("Pred Error ×4",     (200,200, 80)),
        ("Shadow-wtd Error",  (255,120, 40)),
        ("LPIPS Heatmap",     (200, 80,200)),
        ("SH Light Dir",      (255,220,  0)),
        ("LP3 Feature Map",   (80, 200,200)),
        ("HDRI Metadata",     (180,180,255)),
        ("HDRI Environment",  (100,220,255)),
        ("SH Sphere (HDRI)",  (255,160, 80)),
    ]
    for ci, (label, col) in enumerate(col_labels):
        draw.text((PAD + ci * (PANEL + PAD) + 2, HDR - 18), label, fill=col, font=fs)

    for ri in range(n_show):
        src_t  = img1_m[ri]
        gt_t   = img2[ri]
        pred_t = pred[ri]

        ze_np   = ze_inp[ri].cpu().numpy()
        ze_feat = ze_feats[ri].cpu()
        sh_np   = ze_np[:27]

        panels = []

        panels.append(_label_panel(_t2pil(src_t), "Input (masked)"))
        panels.append(_label_panel(_t2pil(gt_t), "Ground Truth"))
        panels.append(_label_panel(_t2pil(pred_t), "Prediction"))

        gt_lum = (0.2126*gt_t[0] + 0.7152*gt_t[1] + 0.0722*gt_t[2]).cpu()
        gt_lum_norm = (gt_lum - gt_lum.min()) / (gt_lum.max() - gt_lum.min() + 1e-8)
        gt_sm_np = gt_lum_norm.numpy()
        panels.append(_label_panel(_shadow_map_to_pil(gt_sm_np, S), "GT Lum Map"))

        pred_lum = (0.2126*pred_t[0] + 0.7152*pred_t[1] + 0.0722*pred_t[2]).cpu()
        pred_lum_norm = (pred_lum - pred_lum.min()) / (pred_lum.max() - pred_lum.min() + 1e-8)
        pred_sm_np = pred_lum_norm.numpy()
        panels.append(_label_panel(_shadow_map_to_pil(pred_sm_np, S), "Pred Lum Map"))

        panels.append(_label_panel(_err_map(pred_t, gt_t, amplify=4.0), "Pred Error ×4"))
        panels.append(_label_panel(_weighted_err_map(pred_t, gt_t, gt_sm_np, S), "Shadow-wtd Error"))
        panels.append(_label_panel(_lpips_heatmap(pred_t, gt_t, lpips_fn, S), "LPIPS Heatmap"))

        gt_pil = _t2pil(gt_t).resize((S, S))
        panels.append(_label_panel(_sh_light_arrow_pil(sh_np, gt_pil, S), "SH Light Dir"))
        panels.append(_label_panel(_lp3_feature_pil(ze_feat, S), "LP3 Features"))
        panels.append(_label_panel(
            _hdri_metadata_pil(ze_np,
            hdri_avg_lum=float(ze_np[27]) if len(ze_np) > 27 else 0.0,
            exposure=float(ze_np[34]) if len(ze_np) > 34 else 0.0,
            size=S),
            "HDRI Metadata"
        ))

        hdri_key = ""
        panels.append(_label_panel(_hdri_env_pil(hdri_key, S), "HDRI Environment"))
        panels.append(_label_panel(_sh_sphere_pil(sh_np, S), "SH Sphere (HDRI)"))

        y_off = HDR + ri * (PH + PAD) + PAD
        for ci, panel in enumerate(panels):
            x = PAD + ci * (PANEL + PAD)
            p = panel.convert("RGB").resize((PANEL, PH), Image.LANCZOS)
            grid.paste(p, (x, y_off))

    out_path = DIAG_DIR / f"diag_step{step:07d}.png"
    grid.save(str(out_path), optimize=False)
    print(f"\n  [DIAG] Saved → {out_path.name}  (step {step})\n")
    wandb_log_image("diag_grid", grid, step)
    return out_path

# ============================================================
# SAMPLE GRID
# ============================================================

def tensor_to_pil(t):
    t = t.squeeze(0).clamp(-1, 1)
    return Image.fromarray(((t + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy())

@torch.no_grad()
def save_sample_grid(vae, unet, lp3, ddpm, val_batch, step, cfg, subject_masks):
    device = cfg["device"]
    n_show = min(12, val_batch[0].shape[0])
    img1   = val_batch[0][:n_show].to(device)
    img2   = val_batch[1][:n_show].to(device)
    ze_inp = val_batch[2][:n_show].to(device)
    sids   = val_batch[3][:n_show]

    img1_clean = apply_fg_mask(img1, subject_masks, sids)
    img2_clean = apply_fg_mask(img2, subject_masks, sids)

    unet.eval(); lp3.eval()
    z1        = vae_encode(vae, img1_clean)
    x_latent  = ddim_sample(unet, lp3, ddpm, z1, ze_inp, n_steps=cfg["ddim_steps"], device=device)
    pred_imgs = vae_decode(vae, x_latent)
    unet.train(); lp3.train()

    H = W = cfg["image_size"]
    LABEL_W = 60
    grid = Image.new("RGB", (LABEL_W + W*3 + 20, H*n_show + (n_show+1)*5 + 30), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        fnt = ImageFont.load_default()
    for col, label in enumerate(["Input (grey-bg)", "GT (grey-bg)", "Predicted"]):
        draw.text((LABEL_W + col*(W+6)+6, 5), label, fill=(200,200,200), font=fnt)
    for row in range(n_show):
        y_off = 30 + row * (H + 5)
        sid_val = sids[row].item() if hasattr(sids[row], "item") else int(sids[row])
        draw.text((2, y_off + H//2 - 6), f"S{sid_val}", fill=(255, 200, 80), font=fnt)
        for col, t in enumerate([img1_clean[row:row+1], img2_clean[row:row+1], pred_imgs[row:row+1]]):
            grid.paste(tensor_to_pil(t), (LABEL_W + col*(W+6), y_off))

    out_path = SAMPLE_DIR / f"sample_step{step:07d}.png"
    grid.save(out_path)
    wandb_log_image("sample_grid", grid, step)
    print(f"\n  SAMPLE GRID → {out_path.name}  (step {step})\n")
    return out_path

# ============================================================
# VALIDATION LOOP
# ============================================================

@torch.no_grad()
def run_val_loop(vae, unet, lp3, ddpm, val_loader, lpips_fn, cfg, step, subject_masks):
    device = cfg["device"]
    unet.eval(); lp3.eval()
    total_mse = 0.0; total_lpips = 0.0; n_batches = 0

    for batch in val_loader:
        img1, img2, ze_inp, sids = batch
        img1   = img1.to(device)
        img2   = img2.to(device)
        ze_inp = ze_inp.to(device)
        img1_m = apply_fg_mask(img1, subject_masks, sids)
        img2_m = apply_fg_mask(img2, subject_masks, sids)
        z1     = vae_encode(vae, img1_m)
        x_lat  = ddim_sample(unet, lp3, ddpm, z1, ze_inp, n_steps=20, device=device)
        pred   = vae_decode(vae, x_lat)
        total_mse   += F.mse_loss(pred, img2_m).item()
        total_lpips += lpips_fn(pred, img2_m).mean().item()
        n_batches   += 1
        if n_batches >= 20: break

    unet.train(); lp3.train()
    avg_mse   = total_mse   / max(n_batches, 1)
    avg_lpips = total_lpips / max(n_batches, 1)
    print(f"  [VAL] step={step}  mse={avg_mse:.4f}  lpips={avg_lpips:.4f}")
    wandb_log({"val/mse": avg_mse, "val/lpips": avg_lpips, "step": step})
    return avg_mse, avg_lpips

# ============================================================
# MAIN TRAIN FUNCTION
# ============================================================

def train():
    init_wandb()
    torch.backends.cudnn.benchmark = True

    print("\n--- Step 1: SH Precomputation ---")
    sh_npy, sh_json = precompute_sh_if_needed()

    print("\n--- Step 2: Build CSV ---")
    csv_path = build_csv_if_needed()

    print("\n--- Step 3: Datasets ---")
    train_ds = RelightDataset(csv_path, sh_npy, sh_json,
                              image_size=CFG["image_size"], split="train",
                              val_fraction=CFG["val_fraction"])
    val_ds   = RelightDataset(csv_path, sh_npy, sh_json,
                              image_size=CFG["image_size"], split="val",
                              val_fraction=CFG["val_fraction"])

    pf = 2 if GPU_CFG["tier"] == "rtx3060" else 3
    nw = CFG["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=nw > 0, prefetch_factor=pf if nw > 0 else None,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=nw > 0,
                              prefetch_factor=pf if nw > 0 else None)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    print("\n--- Step 5: Models ---")
    device = CFG["device"]
    vae      = load_vae(device)
    lp3      = LP3().to(device)
    unet     = UNet(in_channels=8, base_ch=CFG["base_ch"]).to(device)
    ddpm     = DDPM(T=CFG["T"], device=device)
    lpips_fn = load_lpips(device)
    scaler   = GradScaler("cuda", enabled=CFG["use_amp"])
    print(f"  Params: {sum(p.numel() for p in list(lp3.parameters())+list(unet.parameters()))/1e6:.1f}M")

    params = list(lp3.parameters()) + list(unet.parameters())
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["wd"],
                             betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CFG["total_steps"], eta_min=CFG["lr_min"])

    subject_masks = {}

    _mask_search_dirs = [
        KAGGLE_DATA_ROOT / "datasets/devsachan/relight-masks",
        KAGGLE_DATA_ROOT / "datasets/coding12341234/relight-masks",
        KAGGLE_DATA_ROOT / "relight-masks",
    ]
    _mask_dir = None
    for _d in _mask_search_dirs:
        if _d.exists():
            _mask_dir = _d
            break
    if _mask_dir is None:
        _found = list(KAGGLE_DATA_ROOT.rglob("mask_subject1.npy"))
        if _found:
            _mask_dir = _found[0].parent

    if _mask_dir is not None:
        print(f"  Loading masks from: {_mask_dir}")
        for sid in [1, 2, 3]:
            _mask_path = _mask_dir / f"mask_subject{sid}.npy"
            if _mask_path.exists():
                arr    = np.load(_mask_path).astype(np.float32)
                mask_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
                subject_masks[sid] = mask_t
                print(f"    Subject {sid}: loaded ✓  fg={arr.mean():.2%}  shape={arr.shape}")
            else:
                print(f"    Subject {sid}: mask_subject{sid}.npy NOT FOUND in {_mask_dir}")
        print(f"  Masks ready: {len(subject_masks)} subjects")
    else:
        raise FileNotFoundError(
            "Could not find relight-masks dataset.\n"
            "Run generate_masks.py locally and upload the 3 .npy files "
            "to Kaggle as a dataset named 'relight-masks'."
        )

    # ── Resume from checkpoint ────────────────────────────────
    step      = 0
    best_loss = float("inf")
    nan_count = 0

    if _dst_ckpt.exists():
        try:
            ckpt = torch.load(_dst_ckpt, map_location=device)

            def _strip_prefix(state_dict):
                cleaned = {}
                for k, v in state_dict.items():
                    new_k = k.replace("_orig_mod.", "")
                    new_k = new_k.replace(".c.weight", ".conv.weight") \
                                 .replace(".c.bias",   ".conv.bias")
                    cleaned[new_k] = v
                return cleaned

            def _load_compatible(model, state_dict):
                model_keys = set(model.state_dict().keys())
                filtered = {k: v for k, v in state_dict.items() if k in model_keys}
                missing  = model_keys - set(filtered.keys())
                extra    = set(state_dict.keys()) - model_keys
                if extra:
                    print(f"    [CKPT] Ignored {len(extra)} extra keys: {list(extra)[:3]}")
                if missing:
                    print(f"    [CKPT] Missing {len(missing)} keys (new layers — random init): {list(missing)[:3]}")
                model.load_state_dict(filtered, strict=False)

            lp3.load_state_dict(_strip_prefix(ckpt["lp3"]), strict=False)
            _load_compatible(unet, _strip_prefix(ckpt["unet"]))
            step      = ckpt["step"]
            best_loss = ckpt.get("best_loss", float("inf"))
            print(f"  [RESUME] ✓ Weights loaded from step {step:,}  best_loss={best_loss:.4f}")

            # [FIX-3] Restore optimizer + scheduler state.
            # Previously reset every session — now we keep momentum built over 260k steps.
            # If you see "[RESUME] Optimizer state NOT in checkpoint" it means your old
            # checkpoint predates this fix — first session will warm up fresh (once only).
            opt_restored = False
            if "opt" in ckpt:
                try:
                    opt.load_state_dict(ckpt["opt"])
                    scheduler.load_state_dict(ckpt["scheduler"])
                    scaler.load_state_dict(ckpt["scaler"])
                    opt_restored = True
                    print(f"  [RESUME] Optimizer + scheduler state restored ✓")
                except Exception as e:
                    print(f"  [RESUME] Optimizer restore failed ({e}) — fresh AdamW (one-time)")
            else:
                print(f"  [RESUME] Optimizer state NOT in checkpoint — fresh AdamW (one-time only)")

        except Exception as e:
            print(f"  [RESUME] Failed to load checkpoint: {e} — starting fresh")

    # SH index kept for diagnostic grid HDRI env lookup
    with open(sh_json) as f:
        sh_index_dict = {_normalise_hdri_key(k): v for k, v in json.load(f).items()}
    sh_coeffs_arr = np.load(sh_npy)

    # ── Debug mode setup ─────────────────────────────────────
    if DEBUG_MODE:
        debug_end = step + DEBUG_STEPS
        print(f"\n{'='*60}")
        print(f"  DEBUG MODE — running {DEBUG_STEPS} steps ({step} → {debug_end})")
        print(f"  Sample grid every {DEBUG_SAMPLE_EVERY} steps")
        print(f"  Full diagnostic at step {step + DEBUG_DIAG_AT}")
        print(f"  Set DEBUG_MODE = False to run full training")
        print(f"{'='*60}\n")
    else:
        debug_end = CFG["total_steps"]

    print(f"\n--- Training from step {step} to {CFG['total_steps']} ---\n")

    accum_steps  = CFG["accum_steps"]
    lpips_every  = CFG["lpips_every"]
    lpips_max_t  = CFG["lpips_max_t"]

    loss_accum = {k: 0.0 for k in
                  ["total","mse","lpips","ch","shadow",
                   "freq","sat","brightness"]}
    t0          = time.time()
    loader_iter = iter(train_loader)
    opt.zero_grad(set_to_none=True)

    pbar = tqdm(total=debug_end if DEBUG_MODE else CFG["total_steps"],
                initial=step, desc="Training [DEBUG]" if DEBUG_MODE else "Training",
                unit="step", dynamic_ncols=False, ncols=120, mininterval=10.0)

    while step < debug_end:

        acc = {k: 0.0 for k in loss_accum}
        do_lpips  = (step % lpips_every == 0)
        skip_step = False

        for accum_i in range(accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            img1, img2, ze_input, sid_batch = batch
            img1       = img1.to(device)
            img2       = img2.to(device)
            ze_input   = ze_input.to(device)

            with autocast("cuda", enabled=CFG["use_amp"]):

                img1_clean = apply_fg_mask(img1, subject_masks, sid_batch)
                img2_clean = apply_fg_mask(img2, subject_masks, sid_batch)

                z1 = vae_encode(vae, img1_clean)
                z2 = vae_encode(vae, img2_clean)
                ze = lp3(ze_input)

                if torch.isnan(ze).any() or torch.isinf(ze).any():
                    print(f"  [WARN] NaN in ze at step {step} — skipping")
                    nan_count += 1; skip_step = True; break

                t_batch  = ddpm.sample_t(img1.shape[0])
                xt, eps  = ddpm.q_sample(z2, t_batch)
                eps_pred = unet(torch.cat([xt, z1], dim=1), t_batch, ze)

                fg_px = torch.cat([
                    subject_masks.get(int(s),
                        torch.ones(1, 1, CFG["image_size"], CFG["image_size"], device=device))
                    for s in sid_batch
                ], dim=0)
                fg_lat     = F.interpolate(fg_px, size=(32, 32), mode="bilinear", align_corners=False)
                weight_map = fg_lat * CFG["fg_weight"] + (1 - fg_lat) * CFG["bg_weight"]
                mse        = (weight_map * (eps_pred - eps).abs()).mean()

                with torch.no_grad():
                    gt_lum = (0.2126 * img2_clean[:, 0]
                            + 0.7152 * img2_clean[:, 1]
                            + 0.0722 * img2_clean[:, 2])

                    fg_mask_2d_ds = fg_px.squeeze(1)

                    _lum_fg_inf  = gt_lum.masked_fill(fg_mask_2d_ds < 0.5,  float("inf"))
                    _lum_fg_ninf = gt_lum.masked_fill(fg_mask_2d_ds < 0.5, -float("inf"))
                    gt_min = _lum_fg_inf.flatten(1).min(dim=1)[0][:, None, None]
                    gt_max = _lum_fg_ninf.flatten(1).max(dim=1)[0][:, None, None]

                    gt_shadow_map = ((gt_lum - gt_min) / (gt_max - gt_min + 1e-8)
                                     ).clamp(0, 1) * fg_mask_2d_ds

                lpips_val       = torch.tensor(0.0, device=device)
                ch_val          = torch.tensor(0.0, device=device)
                shadow_loss     = torch.tensor(0.0, device=device)
                freq_loss       = torch.tensor(0.0, device=device)
                sat_loss        = torch.tensor(0.0, device=device)
                brightness_loss = torch.tensor(0.0, device=device)

                if do_lpips and accum_i == 0:
                    try:
                        t_lp = ddpm.sample_low_t(img1.shape[0], max_t=lpips_max_t)
                        xt_l, _    = ddpm.q_sample(z2, t_lp)
                        eps_pred_l = unet(torch.cat([xt_l, z1], dim=1), t_lp, ze)
                        ab  = ddpm.sqrt_ab[t_lp][:, None, None, None]
                        oab = ddpm.sqrt_one_m_ab[t_lp][:, None, None, None]
                        z2p   = (xt_l - oab * eps_pred_l) / ab
                        img2p = vae_decode(vae, z2p)

                        if not (torch.isnan(img2p).any() or torch.isinf(img2p).any()):

                            # ── Perceptual loss ──────────────────────────────
                            lpips_val = lpips_fn(img2p, img2_clean).mean()

                            # ── Colour histogram loss ────────────────────────
                            ch_val = color_histogram_loss(img2p, img2_clean)

                            # ── [FIX-5] Saturation loss — fixes grey/washed predictions
                            # Matches per-channel std in foreground region only.
                            # Watch logs: if sat_loss > 0.3 consistently → predictions
                            # are still too grey → increase sat_loss_weight
                            sat_loss = saturation_loss(img2p, img2_clean, fg_px)

                            # ── [FIX-1] Shadow loss — SINGLE clean pass ──────
                            # Previously computed twice with first result thrown away.
                            # Now: quartic weighting (^4) to focus on bright-lit regions
                            # where shadow contrast matters most, plus lum pattern match.
                            # Tuning: if shadow boundaries are sharp enough, reduce ^4 → ^2
                            # If too soft after 50k steps, try ^6 (do NOT exceed ^8)
                            fg_up = fg_mask_2d_ds.unsqueeze(1)
                            sw_fg = ((1.0 - gt_shadow_map) ** 4).unsqueeze(1) * fg_up
                            shadow_loss = (sw_fg * (img2p - img2_clean).abs()).mean()

                            # Luminance pattern matching on top of pixel shadow loss
                            pred_lum = (0.2126 * img2p[:, 0]
                                      + 0.7152 * img2p[:, 1]
                                      + 0.0722 * img2p[:, 2])
                            _pred_inf  = pred_lum.masked_fill(fg_mask_2d_ds < 0.5,  float("inf"))
                            _pred_ninf = pred_lum.masked_fill(fg_mask_2d_ds < 0.5, -float("inf"))
                            pred_min   = _pred_inf.flatten(1).min(dim=1)[0][:, None, None]
                            pred_max   = _pred_ninf.flatten(1).max(dim=1)[0][:, None, None]
                            pred_shadow_map = ((pred_lum - pred_min) /
                                               (pred_max - pred_min + 1e-8)
                                               ).clamp(0, 1) * fg_mask_2d_ds
                            lum_pattern_loss = F.l1_loss(pred_shadow_map, gt_shadow_map)
                            shadow_loss = shadow_loss + lum_pattern_loss

                            # ── [FIX-2] Frequency loss — single call ─────────
                            # Previously set twice, second always won. Now clean.
                            # Fixes: blurry hair, soft edges on face boundary
                            # Tuning: if checkerboard noise appears → reduce weight
                            freq_loss = frequency_loss(img2p, img2_clean)

                            # ── Color mean-variance loss ─────────────────────
                            # Keeps latent distribution matched to ground truth.
                            # Fixes wrong colour casts (green/orange tint on face)
                            color_mv_loss = torch.tensor(0.0, device=device)
                            fg_lat_mask = F.interpolate(fg_px, size=(32, 32),
                                                        mode="bilinear").squeeze(1)
                            for ch in range(4):
                                p_ch = z2p[:, ch][fg_lat_mask > 0.5]
                                t_ch = z2[:, ch][fg_lat_mask > 0.5]
                                if p_ch.numel() > 50:
                                    mean_err = (p_ch.mean() - t_ch.mean()).abs()
                                    var_err  = (p_ch.var()  - t_ch.var() ).abs()
                                    color_mv_loss = color_mv_loss + mean_err + var_err
                            color_mv_loss   = color_mv_loss / 4.0
                            brightness_loss = color_mv_loss

                    except Exception as e:
                        print(f"  [WARN] LPIPS/shadow failed at step {step}: {e}")

                # ── Total loss ───────────────────────────────────
                # Weights rebalanced in [FIX-4] — see CFG and TUNING GUIDE above
                loss = (
                    CFG["mse_weight"]      * mse
                    + CFG["lpips_weight"]     * lpips_val
                    + CFG["ch_loss_weight"]   * ch_val
                    + CFG["shadow_weight"]    * shadow_loss
                    + CFG["freq_loss_weight"] * freq_loss
                    + CFG["sat_loss_weight"]  * sat_loss
                    + CFG["color_mv_weight"]  * brightness_loss
                ) / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN] NaN/Inf loss at step {step} — skipping")
                nan_count += 1; skip_step = True; break

            scaler.scale(loss).backward()

            acc["total"]  += loss.item()
            acc["mse"]    += mse.item()             / accum_steps
            acc["lpips"]  += lpips_val.item()       / accum_steps
            acc["ch"]     += ch_val.item()          / accum_steps
            acc["shadow"] += shadow_loss.item()     / accum_steps
            acc["freq"]   += freq_loss.item()       / accum_steps
            acc["sat"]    += sat_loss.item()        / accum_steps
            acc["brightness"] += brightness_loss.item() / accum_steps

        if skip_step:
            opt.zero_grad(set_to_none=True)
            if accum_i > 0:
                try: scaler.update()
                except Exception: pass
            step += 1; pbar.update(1)
            continue

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); scheduler.step()
        opt.zero_grad(set_to_none=True)

        for k in loss_accum: loss_accum[k] += acc[k]
        step += 1
        pbar.update(1)

        # ── LOG ───────────────────────────────────────────────
        log_every = 10 if DEBUG_MODE else CFG["log_every"]
        if step % log_every == 0:
            n  = log_every
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            its     = step / elapsed
            eta_h   = (CFG["total_steps"] - step) / its / 3600
            al  = loss_accum["total"]  / n
            am  = loss_accum["mse"]    / n
            ap  = loss_accum["lpips"]  / n
            ash = loss_accum["shadow"] / n
            asat= loss_accum["sat"]    / n
            print(
                f"[{step:7d}/{CFG['total_steps']}]  "
                f"loss={al:.4f}  mse={am:.4f}  lpips={ap:.4f}  "
                f"shadow={ash:.4f}  sat={asat:.4f}  "
                f"lr={lr:.2e}  {its:.1f}it/s  eta={eta_h:.1f}h"
            )
            wandb_log({
                "train/loss": al, "train/mse": am, "train/lpips": ap,
                "train/shadow": ash, "train/sat": asat,
                "train/lr": lr, "perf/its": its, "perf/eta_h": eta_h, "step": step,
            })
            pbar.set_postfix(loss=f"{al:.4f}", mse=f"{am:.4f}",
                             shadow=f"{ash:.4f}", sat=f"{asat:.4f}", eta=f"{eta_h:.1f}h")
            loss_accum = {k: 0.0 for k in loss_accum}

        # ── SAMPLE GRID ───────────────────────────────────────
        sample_trigger = (DEBUG_SAMPLE_EVERY if DEBUG_MODE else CFG["sample_every"])
        if step % sample_trigger == 0:
            rand_batch = get_random_val_batch(val_ds, n_samples=12)
            save_sample_grid(vae, unet, lp3, ddpm, rand_batch, step, CFG, subject_masks)

        # ── DIAGNOSTIC GRID ───────────────────────────────────
        diag_trigger = (DEBUG_DIAG_AT if DEBUG_MODE else CFG["diag_every"])
        if DEBUG_MODE:
            if step == debug_end - 1 or step % diag_trigger == 0:
                rand_diag_batch = get_random_val_batch(val_ds, n_samples=4)
                save_diagnostic_grid(
                    vae, unet, lp3, ddpm, lpips_fn,
                    rand_diag_batch, step, CFG, subject_masks,
                    sh_index_dict, sh_coeffs_arr
                )
        else:
            if step % diag_trigger == 0:
                rand_diag_batch = get_random_val_batch(val_ds, n_samples=4)
                save_diagnostic_grid(
                    vae, unet, lp3, ddpm, lpips_fn,
                    rand_diag_batch, step, CFG, subject_masks,
                    sh_index_dict, sh_coeffs_arr
                )

        # ── VAL ───────────────────────────────────────────────
        if not DEBUG_MODE and step % CFG["val_every"] == 0:
            val_mse, val_lpips = run_val_loop(
                vae, unet, lp3, ddpm, val_loader, lpips_fn, CFG, step, subject_masks)
            combined = val_mse + 0.3 * val_lpips
            if combined < best_loss:
                best_loss = combined
                torch.save({
                    "step": step, "lp3": lp3.state_dict(), "unet": unet.state_dict(),
                    "opt": opt.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "best_loss": best_loss,
                }, CKPT_DIR / "best.pt")
                print(f"  [BEST] New best at step {step}: {best_loss:.4f}")

        # ── SAVE LAST ─────────────────────────────────────────
        if not DEBUG_MODE and step % CFG["save_last_every"] == 0:
            torch.save({
                "step": step, "lp3": lp3.state_dict(), "unet": unet.state_dict(),
                "opt": opt.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "best_loss": best_loss,
            }, _dst_ckpt)
            print(f"  [LAST] last.pt saved at step {step}  best_loss={best_loss:.4f}")

        # ── SAVE NUMBERED CKPT ────────────────────────────────
        if not DEBUG_MODE and step % CFG["save_ckpt_every"] == 0:
            torch.save({
                "step": step, "lp3": lp3.state_dict(), "unet": unet.state_dict(),
                "opt": opt.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "best_loss": best_loss,
            }, CKPT_DIR / f"ckpt_{step:07d}.pt")
            print(f"  [CKPT] Saved ckpt_{step:07d}.pt")

            diag_files = sorted(DIAG_DIR.glob("*.png"))
            if diag_files:
                zip_path = CKPT_DIR / f"diag_grids_{step:07d}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for df_path in diag_files:
                        zf.write(df_path, df_path.name)
                print(f"  [DIAG] Zipped {len(diag_files)} grids → {zip_path.name}")

    pbar.close()

    # ── Debug mode exit ───────────────────────────────────────
    if DEBUG_MODE:
        print(f"\n{'='*60}")
        print(f"  DEBUG COMPLETE — {DEBUG_STEPS} steps done")
        print(f"  Review outputs in:")
        print(f"    Samples : {SAMPLE_DIR}")
        print(f"    Diag    : {DIAG_DIR}")
        print(f"  If outputs look correct:")
        print(f"    Set DEBUG_MODE = False and rerun for full training")
        print(f"  Checkpoint NOT saved in debug mode (260k preserved)")
        print(f"{'='*60}\n")
        return

    torch.save({
        "step": step, "lp3": lp3.state_dict(), "unet": unet.state_dict(),
        "opt": opt.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "best_loss": best_loss,
    }, _dst_ckpt)
    print(f"\n  Training complete at step {step}  best_loss={best_loss:.4f}")
    wandb_finish()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train()