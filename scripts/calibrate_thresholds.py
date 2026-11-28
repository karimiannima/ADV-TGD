"""Calibrate FAR thresholds for this pipeline's preprocessing.

The thresholds in config.THRESHOLDS only fit FaceNet on whole-image input. This script aligns CelebA-HQ images exactly
like main.py (FaceProcessor.align_face), embeds them with every FR model and takes the quantiles of the different-person
cosine scores (CelebA-HQ images are almost all different people) at FAR 0.1 / 0.01 / 0.001. It does this for several
centre-crop fractions of the aligned image (1.0 = what main.py feeds the FR models by default).

Run from the project root:
    python scripts/calibrate_thresholds.py [--n 300] [--export configs/thresholds_crop1.json --export-crop 1.0]
"""
import argparse
import glob
import json
import os
import sys

import face_alignment
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import AdvTGDConfig, FR_INPUT_SIZE, THRESHOLDS
from face_processor import FaceProcessor
from utils import get_fr_model

FARS = (0.1, 0.01, 0.001)
CROPS = (1.0, 0.85, 0.7, 0.6, 0.5)
BATCH = 20


def load_aligned(proc, folder, n):
    """Aligned, normalised tensors of the first n images in `folder` where a face was found."""
    imgs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.jpg")))[:2 * n]:
        im, _ = proc.align_face(path)
        if im is not None:
            imgs.append(proc.tfm(im).unsqueeze(0))
        if len(imgs) == n:
            break
    return torch.cat(imgs)


@torch.no_grad()
def embed(model, x, size):
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return F.normalize(model(x), dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="number of images (pairs = n*(n-1)/2)")
    ap.add_argument("--images", default="assets/datasets/CelebA-HQ", help="folder of .jpg images")
    ap.add_argument("--out", default="results_analysis/threshold_check.json")
    ap.add_argument("--export", help="also write a thresholds file usable as ADV_THRESH_JSON")
    ap.add_argument("--export-crop", type=float, default=1.0, choices=CROPS)
    args = ap.parse_args()

    cfg = AdvTGDConfig()
    dev = cfg.device
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=str(dev))
    x = load_aligned(FaceProcessor(fa, cfg), args.images, args.n).to(dev)
    n_img, side = len(x), x.shape[-1]
    i, j = torch.triu_indices(n_img, n_img, offset=1)
    result = {"n_images": n_img, "n_pairs": int(i.numel()), "crops": {}}

    for name, size in FR_INPUT_SIZE.items():
        model = get_fr_model(name, dev).eval().requires_grad_(False)
        cfg_thr = THRESHOLDS.get(name)
        for crop in CROPS:
            s = int(side * crop)
            o = (side - s) // 2
            xc = x[:, :, o:o + s, o:o + s]
            emb = torch.cat([embed(model, xc[k:k + BATCH], size) for k in range(0, n_img, BATCH)])
            scores = (emb @ emb.T)[i, j]
            entry = {"mean_impostor_cos": float(scores.mean()),
                     "empirical_thresholds": [round(float(torch.quantile(scores, 1 - far)), 4) for far in FARS]}
            if cfg_thr:
                entry["config_thresholds"] = list(cfg_thr)
                entry["frac_above_config"] = [round(float((scores > t).float().mean()), 4) for t in cfg_thr]
            result["crops"].setdefault(str(crop), {})[name] = entry
            print(f"crop={crop:<4} {name:10s} mean_cos={entry['mean_impostor_cos']:.3f} "
                  f"frac_above_config={entry.get('frac_above_config')} empirical_thr={entry['empirical_thresholds']}",
                  flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    if args.export:
        chosen = result["crops"][str(args.export_crop)]
        os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
        with open(args.export, "w") as f:
            json.dump({name: v["empirical_thresholds"] for name, v in chosen.items()}, f, indent=1)


if __name__ == "__main__":
    main()
