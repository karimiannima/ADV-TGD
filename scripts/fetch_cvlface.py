"""Download the ArcFace and AdaFace (IR101, WebFace4M) surrogate checkpoints into pretrained_model/. Public, no token needed.
Run from the project root: python scripts/fetch_cvlface.py"""
from huggingface_hub import snapshot_download

for repo, dst in (("minchul/cvlface_arcface_ir101_webface4m", "pretrained_model/arcface_ir101"),
                  ("minchul/cvlface_adaface_ir101_webface4m", "pretrained_model/adaface_ir101")):
    snapshot_download(repo, local_dir=dst)
    print("ok", repo, "->", dst)
