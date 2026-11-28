"""Build pairs_manifest_updated.json from a sample folder so main.py runs in Manifest Mode
(round-robin src -> target pairing, with the per-target prompt from <img_dir>/target_text/<target>.txt).

    python make_manifest.py                       # celeba-hq_sample -> pairs_manifest_updated.json
    python make_manifest.py --img-dir my_faces    # my_faces/src, my_faces/target, my_faces/target_text
"""
import os, json, argparse

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

ap = argparse.ArgumentParser()
ap.add_argument("--img-dir", default="celeba-hq_sample")
ap.add_argument("--out", default="pairs_manifest_updated.json")
ap.add_argument("--default-text", default="portrait photo")
args = ap.parse_args()

src_dir, tgt_dir, txt_dir = (os.path.join(args.img_dir, d) for d in ("src", "target", "target_text"))
srcs = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS))
tgts = sorted(f for f in os.listdir(tgt_dir) if f.lower().endswith(IMG_EXTS))
assert srcs and tgts, f"no images under {src_dir} / {tgt_dir}"

pairs = []
for i, s in enumerate(srcs):
    t = tgts[i % len(tgts)]
    txt_path = os.path.join(txt_dir, os.path.splitext(t)[0] + ".txt")
    text = open(txt_path, encoding="utf-8", errors="ignore").read().strip() if os.path.exists(txt_path) else ""
    pairs.append({
        "pair_id": f"{os.path.splitext(s)[0]}__to__{os.path.splitext(t)[0]}",
        "src_path": os.path.join(src_dir, s).replace("\\", "/"),
        "tgt_path": os.path.join(tgt_dir, t).replace("\\", "/"),
        "text": text or args.default_text,
    })

with open(args.out, "w", encoding="utf-8") as f:
    json.dump({"pairs": pairs}, f, indent=2)
print(f"wrote {args.out} with {len(pairs)} pairs")
for p in pairs:
    print(f"  {p['pair_id']}: {p['text'][:70]!r}...")
