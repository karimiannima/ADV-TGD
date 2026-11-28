"""FID of the attacked faces against the clean source faces, for one or more main.py run folders.

Needs `pytorch-fid` (its Inception weights are cached under TORCH_HOME). With few pairs (25) the 2048-d statistic is
rough: compare runs on the same pairs, do not read the absolute value.

Run from the project root:
    python scripts/compute_fid.py results_X/run_Adv_TGD_Main [more run folders ...]
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

SOURCE = "src_aligned.png"
ATTACKED = ("final_best_REEVALUATED.png", "final_best_blended_REEVALUATED.png")
INCEPTION_SIZE = 299
FEATURE_DIM = 2048


def load_inception(device):
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]]).to(device).eval()


@torch.no_grad()
def statistics(net, paths, device):
    """Mean and covariance of the Inception features of the images."""
    feats = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((INCEPTION_SIZE, INCEPTION_SIZE))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1)[None].to(device)
        feats.append(net(x)[0].flatten(1).cpu().numpy())
    feats = np.concatenate(feats)
    return feats.mean(0), np.cov(feats, rowvar=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="main.py run folders")
    ap.add_argument("--out", help="optional JSON file for the numbers")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_inception(device)
    results = {}
    for run in args.runs:
        pairs = [d for d in sorted(glob.glob(os.path.join(run, "*__to__*"))) if os.path.isfile(os.path.join(d, SOURCE))]
        if len(pairs) < 2:
            print(f"{run}: skipped, {len(pairs)} pair folders")
            continue
        mu_s, sigma_s = statistics(net, [os.path.join(d, SOURCE) for d in pairs], device)
        for name in ATTACKED:
            found = [d for d in pairs if os.path.isfile(os.path.join(d, name))]
            if len(found) < 2:
                print(f"{run}  {name}: skipped, {len(found)} files")
                continue
            mu_a, sigma_a = statistics(net, [os.path.join(d, name) for d in found], device)
            fid = float(calculate_frechet_distance(mu_s, sigma_s, mu_a, sigma_a))
            results.setdefault(run, {})[name] = {"n": len(found), "fid_vs_source": round(fid, 2)}
            print(f"{run}  {name}  n={len(found)}  FID_vs_source={fid:.2f}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
