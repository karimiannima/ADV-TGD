"""Multi-query black-box transfer attack: pick the attack image by querying the UNSEEN face recognition model.

The training loop only ever sees the white-box surrogates (FaceNet, IRSE50). Scoring a single image against a
held-out model therefore measures one lucky draw. A real attacker against a deployed verifier can submit
several candidates and keep whichever is accepted. This script runs that attacker: it takes the step images
from --start to the last step as the candidate pool, queries a held-out model with each, and counts the pair
as a success if ANY candidate is accepted at the given false accept rate.

    python query_attack.py --results results_ccs_eval/run_Adv_TGD_Main                    # steps 40..last, IR152 + MobileFace
    python query_attack.py --results ... --start 40 --far 0.01 --csv query_scores.csv
    python query_attack.py --results ... --frs IR152 --plot query_budget.png              # ASR vs query budget

WHY THE CONTROL MATTERS. Taking the best of N noisy scores is biased upward even when nothing is happening,
so a 30-query attack must not be compared against a 1-query baseline. Every run therefore also reports a
MISMATCHED-TARGET control: the same candidate images, the same budget, the same take-the-best rule, but
scored against a different identity from the target set. Whatever that control earns is what the selection
procedure alone buys. Report the gap, never the raw number.

Threat model: the attacker needs accept/reject feedback from the deployed system for each query (a credential
stuffing style oracle). A system that rate-limits or logs repeated verification attempts breaks this attack
even when a single-image attack would also have failed; that is one of the defences worth discussing.
"""
import os, re, glob, argparse, csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import THRESHOLDS
from utils import get_fr_model

FAR_INDEX = {0.1: 0, 0.01: 1, 0.001: 2}
_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


def fr_size(name):
    return 112 if name in ("IR152", "IRSE50", "MobileFace") else 160


def load(path, res=768):
    img = Image.open(path).convert("RGB")
    if img.size != (res, res):
        img = img.resize((res, res), Image.BILINEAR)
    return _tfm(img).unsqueeze(0)


@torch.no_grad()
def embed(model, x, size):
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return F.normalize(model(x), p=2, dim=1)


def step_of(path):
    return int(re.search(r"step_(\d+)", os.path.basename(path)).group(1))


def evenly_spaced(items, k):
    """k candidates spread over the pool (k=1 -> the last step, the single-image case)."""
    if k >= len(items):
        return list(items)
    if k == 1:
        return [items[-1]]
    idx = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in sorted(set(idx))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="e.g. results_ccs_eval/run_Adv_TGD_Main")
    ap.add_argument("--start", type=int, default=40, help="first step of the candidate pool (default 40)")
    ap.add_argument("--end", type=int, default=None, help="last step of the pool (default: the last step present)")
    ap.add_argument("--frs", nargs="+", default=["IR152", "MobileFace"],
                    help="models to query; these should be the HELD-OUT ones")
    ap.add_argument("--far", type=float, default=0.01, choices=[0.1, 0.01, 0.001])
    ap.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 5, 10, 20, 30],
                    help="query budgets to report (1 = the single-image case)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv", default=None, help="write the per-pair, per-model detail here")
    ap.add_argument("--plot", default=None, help="save an ASR-vs-budget PNG here (needs matplotlib)")
    args = ap.parse_args()

    device = torch.device(args.device)
    tau = {n: THRESHOLDS[n][FAR_INDEX[args.far]] for n in args.frs}
    models = {n: get_fr_model(n, device=device).eval() for n in args.frs}

    pairs = sorted(d for d in glob.glob(os.path.join(args.results, "*")) if os.path.isdir(d))
    pairs = [p for p in pairs if os.path.exists(os.path.join(p, "tgt_aligned.png"))]
    if not pairs:
        print("no pair folders found"); return

    # target embeddings per pair, plus a mismatched target for the control
    tgt_emb, tgt_path = {}, {}
    for p in pairs:
        tp = os.path.join(p, "tgt_aligned.png")
        tgt_path[p] = tp
        t = load(tp).to(device)
        tgt_emb[p] = {n: embed(m, t, fr_size(n)) for n, m in models.items()}

    # mismatched control: pair i is scored against the target of a pair with a DIFFERENT target image
    import hashlib
    def tgt_id(p):
        return hashlib.md5(open(tgt_path[p], "rb").read()).hexdigest()[:8]
    ids = {p: tgt_id(p) for p in pairs}
    control_of = {}
    for p in pairs:
        others = [q for q in pairs if ids[q] != ids[p]]
        control_of[p] = others[hash(p) % len(others)] if others else None

    rows, per_model = [], {n: {"hit": {}, "ctrl": {}, "first": [], "npass": []} for n in args.frs}
    for n in args.frs:
        for k in args.budgets:
            per_model[n]["hit"][k] = 0
            per_model[n]["ctrl"][k] = 0

    print(f"candidate pool: steps {args.start}..{args.end if args.end is not None else 'last'}   "
          f"queried models: {', '.join(args.frs)}   FAR {args.far}")
    for p in pairs:
        steps = sorted(glob.glob(os.path.join(p, "step_*.png")), key=step_of)
        pool = [s for s in steps if step_of(s) >= args.start and (args.end is None or step_of(s) <= args.end)]
        if not pool:
            print("skip (no steps in range):", os.path.basename(p)); continue
        imgs = [load(s).to(device) for s in pool]
        line = {}

        for n, m in models.items():
            e = [embed(m, x, fr_size(n)) for x in imgs]
            cos = [float((ei * tgt_emb[p][n]).sum()) for ei in e]
            cpair = control_of[p]
            cos_ctrl = [float((ei * tgt_emb[cpair][n]).sum()) for ei in e] if cpair else []

            passing = [step_of(s) for s, c in zip(pool, cos) if c > tau[n]]
            per_model[n]["npass"].append(len(passing))
            per_model[n]["first"].append(min(passing) if passing else None)
            line[n] = len(passing)

            for k in args.budgets:
                sel = evenly_spaced(list(range(len(pool))), k)
                if any(cos[i] > tau[n] for i in sel):
                    per_model[n]["hit"][k] += 1
                if cos_ctrl and any(cos_ctrl[i] > tau[n] for i in sel):
                    per_model[n]["ctrl"][k] += 1

            best_i = max(range(len(cos)), key=lambda i: cos[i])
            rows.append({
                "pair": os.path.basename(p), "model": n, "far": args.far, "tau": round(tau[n], 6),
                "pool_steps": len(pool), "n_passing": len(passing),
                "first_passing_step": passing[0] if passing else "",
                "best_step": step_of(pool[best_i]), "best_cos": round(cos[best_i], 4),
                "last_step_cos": round(cos[-1], 4),
                "control_best_cos": round(max(cos_ctrl), 4) if cos_ctrl else "",
                "control_pair": os.path.basename(control_of[p]) if control_of[p] else "",
            })
        print(f"  {os.path.basename(p):<22} " +
              "  ".join(f"{n}: {line[n]:>2}/{len(pool)} accepted" for n in args.frs))

    N = len(set(r["pair"] for r in rows))
    print(f"\n{N} pairs, candidate pool of {rows[0]['pool_steps']} steps each, FAR {args.far}\n")
    hdr = f"{'model':<12}{'budget':>7}{'ASR':>8}{'control':>9}{'lift':>8}   {'median first pass':>18}"
    print(hdr); print("-" * len(hdr))
    for n in args.frs:
        firsts = [f for f in per_model[n]["first"] if f is not None]
        med = sorted(firsts)[len(firsts) // 2] if firsts else None
        for k in args.budgets:
            if k > rows[0]["pool_steps"] and k != args.budgets[-1]:
                continue
            a = 100.0 * per_model[n]["hit"][k] / N
            c = 100.0 * per_model[n]["ctrl"][k] / N
            tag = "  (single image)" if k == 1 else ""
            print(f"{n:<12}{min(k, rows[0]['pool_steps']):>7}{a:>7.1f}%{c:>8.1f}%{a - c:>+7.1f}"
                  f"{(f'  step {med}' if k == args.budgets[0] and med is not None else ''):>20}{tag}")
        npass = per_model[n]["npass"]
        print(f"{'':<12}candidates accepted per pair: mean {sum(npass)/len(npass):.1f} of {rows[0]['pool_steps']}, "
              f"{sum(1 for x in npass if x == 0)} pairs never accepted\n")
    print("ASR = any queried candidate accepted. control = same images and budget scored against a different\n"
          "identity, so it measures what take-the-best-of-N buys on its own. Report the lift, not the ASR alone.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", args.csv)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ks = [k for k in args.budgets if k <= rows[0]["pool_steps"]] or args.budgets[:1]
        for n in args.frs:
            ax.plot(ks, [100.0 * per_model[n]["hit"][k] / N for k in ks], "-o", label=f"{n} attack")
            ax.plot(ks, [100.0 * per_model[n]["ctrl"][k] / N for k in ks], "--x",
                    label=f"{n} mismatched-target control")
        ax.set_xlabel(f"query budget (candidates drawn from steps {args.start}..last)")
        ax.set_ylabel(f"pairs accepted at FAR {args.far} (%)")
        ax.set_title("Multi-query transfer against held-out models")
        ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
