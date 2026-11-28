"""Re-score the saved aligned frames of a main.py run with every FR model that has a threshold.

main.py picks the saved step of each pair with the evaluation model's own cosine (allowed by the protocol, but it makes
the reported ASR optimistic). For every FR model and FAR level this script reports the share of pairs above threshold for
  src_vs_tgt : the clean source against the target (no attack)
  last       : the last saved step (no selection)
  best_step  : the best saved step by that model's own cosine
  main_pick  : the frame main.py saved as final_best_REEVALUATED.png

Run from the project root, after main.py wrote results_<run id>/run_Adv_TGD_Main/<pair>/step_*.png:
    ADV_THRESH_JSON=configs/thresholds_crop1.json python scripts/rescore_saved_frames.py results_X/run_Adv_TGD_Main
Set ADV_FR_CROP to the same value that main.py used.
"""
import argparse
import glob
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import FR_INPUT_SIZE, THRESHOLDS
from utils import get_fr_model

FAR_INDEX = {"FAR0.01": 1, "FAR0.001": 2}
METRICS = ("src_vs_tgt", "last", "best_step", "main_pick")
MIN_EVAL_STEP = 10  # steps before this are warm-up and are not eligible for "best_step"
RESOLUTION = 768    # side of the aligned images main.py saves


def step_number(path):
    return int(re.search(r"step_(\d+)", path).group(1))


class Scorer:
    def __init__(self, names, device, crop):
        self.device, self.crop = device, crop
        self.models = {n: get_fr_model(n, device).eval().requires_grad_(False) for n in names}
        self.tfm = transforms.Compose([transforms.Resize((RESOLUTION, RESOLUTION)), transforms.ToTensor(),
                                       transforms.Normalize([.5] * 3, [.5] * 3)])

    @torch.no_grad()
    def embed(self, name, paths):
        x = torch.stack([self.tfm(Image.open(p).convert("RGB")) for p in paths]).to(self.device)
        if self.crop < 1.0:  # same centre crop as main.py's ADV_FR_CROP
            side = int(x.shape[-1] * self.crop)
            off = (x.shape[-1] - side) // 2
            x = x[:, :, off:off + side, off:off + side]
        size = FR_INPUT_SIZE[name]
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return F.normalize(self.models[name](x), dim=1)


def score_pair(scorer, pair_dir, steps, pick):
    candidates = [p for p in steps if step_number(p) >= MIN_EVAL_STEP] or steps
    row = {"pair": os.path.basename(pair_dir)}
    for name in scorer.models:
        tgt = scorer.embed(name, [os.path.join(pair_dir, "tgt_aligned.png")])
        src = scorer.embed(name, [os.path.join(pair_dir, "src_aligned.png")])
        steps_cos = (scorer.embed(name, candidates) @ tgt.T).squeeze(1)
        row[name] = {"src_vs_tgt": float(src @ tgt.T), "last": float(steps_cos[-1]),
                     "best_step": float(steps_cos.max()), "main_pick": float(scorer.embed(name, [pick]) @ tgt.T)}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="main.py run folder, e.g. results_X/run_Adv_TGD_Main")
    ap.add_argument("--out", default="results_analysis/selection_check.json")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scorer = Scorer([n for n in FR_INPUT_SIZE if n in THRESHOLDS], device, float(os.environ.get("ADV_FR_CROP", "1.0")))

    rows = []
    for pair_dir in sorted(glob.glob(os.path.join(args.run, "*"))):
        steps = sorted(glob.glob(os.path.join(pair_dir, "step_*.png")), key=step_number)
        pick = os.path.join(pair_dir, "final_best_REEVALUATED.png")
        if steps and os.path.exists(pick):
            rows.append(score_pair(scorer, pair_dir, steps, pick))

    summary = {f"{n}@{far}": {m: sum(r[n][m] > THRESHOLDS[n][i] for r in rows) / max(1, len(rows)) for m in METRICS}
               for n in scorer.models for far, i in FAR_INDEX.items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"run": args.run, "pairs": rows, "summary": summary}, f, indent=1)

    print(f"pairs={len(rows)}  share of pairs above threshold per model/FAR: " + " | ".join(METRICS))
    for key, values in summary.items():
        print(f"{key:20s} " + " | ".join(f"{v:.2f}" for v in values.values()))


if __name__ == "__main__":
    main()
