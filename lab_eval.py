"""White-box vs held-out black-box evaluation of Adv-TGD results (CPU friendly).

Scores every pair folder under --results with the four FR models and reports, per model, the attack success
rate (ASR) at FAR 0.1 / 0.01 / 0.001 and the mean cosine to the target, split into
  white-box  = models used in the training loss (default: FaceNet, IRSE50)
  black-box  = held-out models never seen by the optimiser (default: IR152, MobileFace)

    python lab_eval.py --results results_ccs_eval/run_Adv_TGD_Main                    # summary over all pairs
    python lab_eval.py --results ... --image step:69                                   # score the last training step
    python lab_eval.py --results ... --trajectory 0__to__1289 --plot traj.png          # cosine vs step for one pair
    python lab_eval.py --results ... --csv lab_scores.csv                              # per-pair table for your report

Images are the 768x768 aligned faces the attack works on (src_aligned.png, tgt_aligned.png and the attack
output); the FR models see them resized to 112x112 (IR152/IRSE50/MobileFace) or 160x160 (FaceNet), exactly
as in main.py.
"""
import os, re, glob, json, argparse, csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import THRESHOLDS
from utils import get_fr_model

FAR_LEVELS = [(0.1, 0), (0.01, 1), (0.001, 2)]
ALL_FRS = ["FaceNet", "IRSE50", "IR152", "MobileFace"]


def fr_size(name):
    return 112 if name in ("IR152", "IRSE50", "MobileFace") else 160


_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


def load(path, res=768):
    img = Image.open(path).convert("RGB")
    if img.size != (res, res):
        img = img.resize((res, res), Image.BILINEAR)
    return _tfm(img).unsqueeze(0)


@torch.no_grad()
def embed(model, x, size):
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return F.normalize(model(x), p=2, dim=1)


def pick_attack_image(pair_dir, which):
    if which == "final":
        for cand in ("final_best_REEVALUATED.png",):
            p = os.path.join(pair_dir, cand)
            if os.path.exists(p):
                return p
        which = "step:last"
    if which == "regen":
        return os.path.join(pair_dir, "regen_from_lora.png")
    if which.startswith("step:"):
        steps = sorted(glob.glob(os.path.join(pair_dir, "step_*.png")))
        if not steps:
            return None
        if which == "step:last":
            return steps[-1]
        n = int(which.split(":")[1])
        p = os.path.join(pair_dir, f"step_{n:06d}.png")
        return p if os.path.exists(p) else None
    return os.path.join(pair_dir, which)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="e.g. results_ccs_eval/run_Adv_TGD_Main")
    ap.add_argument("--image", default="final",
                    help="which attack image to score: final (final_best_REEVALUATED.png, falls back to the last step), "
                         "step:last, step:N, regen (regen_from_lora.png) or a file name inside the pair folder")
    ap.add_argument("--frs", nargs="+", default=ALL_FRS)
    ap.add_argument("--white-box", nargs="+", default=["FaceNet", "IRSE50"], help="models used in the training loss")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv", default=None, help="write the per-pair, per-model scores here")
    ap.add_argument("--trajectory", default=None, help="pair id: score every step_*.png of that pair instead")
    ap.add_argument("--plot", default=None, help="with --trajectory: save a cosine-vs-step PNG here (needs matplotlib)")
    args = ap.parse_args()

    device = torch.device(args.device)
    models = {n: get_fr_model(n, device=device).eval() for n in args.frs}

    # ---------------- trajectory mode ----------------
    if args.trajectory:
        pd = os.path.join(args.results, args.trajectory)
        tgt = load(os.path.join(pd, "tgt_aligned.png")).to(device)
        src = load(os.path.join(pd, "src_aligned.png")).to(device)
        e_t = {n: embed(m, tgt, fr_size(n)) for n, m in models.items()}
        steps = sorted(glob.glob(os.path.join(pd, "step_*.png")),
                       key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
        rows = []
        for p in steps:
            k = int(re.search(r"step_(\d+)", p).group(1))
            x = load(p).to(device)
            rows.append({"step": k, **{n: (embed(m, x, fr_size(n)) * e_t[n]).sum().item() for n, m in models.items()}})
        base = {n: (embed(m, src, fr_size(n)) * e_t[n]).sum().item() for n, m in models.items()}
        print(f"pair {args.trajectory}: cos(src,tgt) baseline " + "  ".join(f"{n}={v:.3f}" for n, v in base.items()))
        print("step  " + "  ".join(f"{n:>10}" for n in models))
        for r in rows:
            if r["step"] % 10 == 0 or r is rows[-1]:
                print(f"{r['step']:>4}  " + "  ".join(f"{r[n]:>10.3f}" for n in models))
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["step"] + list(models)); w.writeheader(); w.writerows(rows)
            print("wrote", args.csv)
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            for n in models:
                style = "-" if n in args.white_box else "--"
                ax.plot([r["step"] for r in rows], [r[n] for r in rows], style,
                        label=f"{n} ({'white-box' if n in args.white_box else 'held-out'})")
                ax.axhline(THRESHOLDS[n][1], lw=0.6, color="grey")
            ax.set_xlabel("LoRA step"); ax.set_ylabel("cos(attack, target)")
            ax.set_title(f"{args.trajectory}: thresholds at FAR 0.01 in grey")
            ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(args.plot, dpi=130)
            print("wrote", args.plot)
        return

    # ---------------- summary mode ----------------
    pairs = sorted(d for d in glob.glob(os.path.join(args.results, "*")) if os.path.isdir(d))
    per_pair = []
    for pd in pairs:
        atk_path = pick_attack_image(pd, args.image)
        if not atk_path or not os.path.exists(atk_path):
            print("skip (no attack image):", pd); continue
        src = load(os.path.join(pd, "src_aligned.png")).to(device)
        tgt = load(os.path.join(pd, "tgt_aligned.png")).to(device)
        atk = load(atk_path).to(device)
        rec = {"pair": os.path.basename(pd), "image": os.path.basename(atk_path)}
        for n, m in models.items():
            s = fr_size(n)
            e_s, e_t, e_a = embed(m, src, s), embed(m, tgt, s), embed(m, atk, s)
            rec[f"{n}:cos_src_tgt"] = (e_s * e_t).sum().item()
            rec[f"{n}:cos_atk_tgt"] = (e_a * e_t).sum().item()
            rec[f"{n}:cos_atk_src"] = (e_a * e_s).sum().item()
        per_pair.append(rec)

    if not per_pair:
        print("no pairs scored"); return
    N = len(per_pair)
    print(f"\n{N} pairs from {args.results}  (attack image: --image {args.image})")
    print(f"{'FR model':<11}{'role':<11}{'ASR@0.1':>9}{'ASR@0.01':>10}{'ASR@0.001':>11}"
          f"{'no-attack@0.01':>16}{'mean cos(atk,tgt)':>19}{'mean cos(src,tgt)':>19}")
    group = {"white-box": [], "black-box": []}
    for n in models:
        role = "white-box" if n in args.white_box else "black-box"
        asr = [100.0 * sum(r[f"{n}:cos_atk_tgt"] > THRESHOLDS[n][idx] for r in per_pair) / N for _, idx in FAR_LEVELS]
        # the same test applied to the UN-attacked source: how often the threshold is already passed with no attack
        base_asr = 100.0 * sum(r[f"{n}:cos_src_tgt"] > THRESHOLDS[n][1] for r in per_pair) / N
        mc = sum(r[f"{n}:cos_atk_tgt"] for r in per_pair) / N
        mb = sum(r[f"{n}:cos_src_tgt"] for r in per_pair) / N
        group[role].append((asr[1], base_asr))
        print(f"{n:<11}{role:<11}{asr[0]:>8.1f}%{asr[1]:>9.1f}%{asr[2]:>10.1f}%{base_asr:>15.1f}%{mc:>19.3f}{mb:>19.3f}")
    for role, v in group.items():
        if v:
            a = sum(x for x, _ in v) / len(v); b = sum(y for _, y in v) / len(v)
            print(f"-> {role}: mean ASR@0.01 {a:.1f}%  vs {b:.1f}% with no attack  (lift {a - b:+.1f} points)")
    print("Read 'no-attack@0.01' first: if the un-attacked source already passes a model's threshold on these crops,"
          " that model's ASR says little; compare the lift, or use FAR 0.001.")
    if args.csv:
        keys = list(per_pair[0].keys())
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per_pair)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
