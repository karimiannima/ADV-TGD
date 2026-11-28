"""CPU-friendly inference from a saved Adv-TGD LoRA adapter.

Re-generates the adversarial face for one trained pair (produced by `main.py --save-lora`) with a single
UNet forward pass + one VAE decode, then scores it against the target identity with the FR models.
No training, so it runs on a CPU in a few minutes.

Example:
    python infer_lora.py --pair-dir results_ccs_eval/run_Adv_TGD_Main/0__to__1289 --device cpu
    python infer_lora.py --pair-dir ... --device cpu --model-id /path/to/stable-diffusion-2-1 --frs IR152 IRSE50 FaceNet MobileFace
"""
import os, json, argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import PeftModel

from config import AdvTGDConfig, THRESHOLDS
from utils import get_fr_model
from losses import psnr, ssim_approx


def fr_size_from_name(name):
    return 112 if name in ("IR152", "IRSE50", "MobileFace") else 160


def load_img(path, res, device):
    tfm = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor(),
                              transforms.Normalize([0.5] * 3, [0.5] * 3)])
    return tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def to_pil(x):
    x = (x[0] * 0.5 + 0.5).clamp(0, 1)
    return Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair-dir", required=True, help="a pair folder that contains lora_adapter/, src_aligned.png, tgt_aligned.png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-id", default=None, help="HF repo id or local diffusers folder (default: config / ADV_TGD_MODEL_ID)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frs", nargs="+", default=["IR152"], help="FR models to score with (needs pretrained_model/*.pth)")
    ap.add_argument("--out", default=None, help="output png (default: <pair-dir>/regen_from_lora.png)")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = AdvTGDConfig()
    if args.model_id:
        cfg.model_id = args.model_id
    pair = args.pair_dir
    summ_path = os.path.join(pair, "train_summary.json")
    summ = json.load(open(summ_path, encoding="utf-8")) if os.path.exists(summ_path) else {}
    res = int(summ.get("resolution", cfg.resolution))
    text = summ.get("text") or "portrait photo"
    t_fixed = summ.get("t_fixed")
    if t_fixed is None:
        t_enc = max(1, int(cfg.strength_train * 1000))
        t_fixed = max(0, min(t_enc - 1, int(cfg.t_fixed_frac * t_enc)))

    print(f"[INFO] device={device} model_id={cfg.model_id} resolution={res} t_fixed={t_fixed} text={text!r}")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model_id, subfolder="text_encoder").eval()
    vae = AutoencoderKL.from_pretrained(cfg.model_id, subfolder="vae").to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(cfg.model_id, subfolder="unet").eval()
    unet = PeftModel.from_pretrained(unet, os.path.join(pair, "lora_adapter")).to(device).eval()
    sched = DDPMScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")
    alphas = sched.alphas_cumprod.to(device)

    src = load_img(os.path.join(pair, "src_aligned.png"), res, device)
    tgt = load_img(os.path.join(pair, "tgt_aligned.png"), res, device)
    mask_path = os.path.join(pair, "mask_hybrid.png")
    latent_mask = None
    if os.path.exists(mask_path):
        m = transforms.ToTensor()(Image.open(mask_path).convert("L")).unsqueeze(0).to(device)
        latent_mask = F.interpolate(m, size=(res // 8, res // 8), mode="area").clamp(0, 1)

    ids = tokenizer([text], padding="max_length", truncation=True, max_length=tokenizer.model_max_length,
                    return_tensors="pt").input_ids
    emb = text_encoder(ids)[0].to(device)

    sf = getattr(vae.config, "scaling_factor", 0.18215)
    lat = vae.encode(src).latent_dist.sample() * sf
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(lat.shape, generator=g).to(device)
    t = torch.full((1,), int(t_fixed), device=device, dtype=torch.long)
    noisy = sched.add_noise(lat, noise, t)
    pred = unet(noisy, t, encoder_hidden_states=emb).sample
    a_t = alphas[t].view(-1, 1, 1, 1)
    x0 = (noisy - (1 - a_t).sqrt() * pred) / a_t.sqrt()           # same epsilon-form x0 estimate as main.py
    if latent_mask is not None:
        x0 = lat * (1 - latent_mask) + x0 * latent_mask
    img = torch.nan_to_num(vae.decode(x0 / sf).sample, nan=0.0).clamp(-1, 1)

    out = args.out or os.path.join(pair, "regen_from_lora.png")
    to_pil(img).save(out)
    print(f"[INFO] wrote {out}")

    rows = []
    for name in args.frs:
        fr = get_fr_model(name, device=device).eval()
        s = fr_size_from_name(name)
        e = lambda x: F.normalize(fr(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)), p=2, dim=1)
        cos_tgt = (e(img) * e(tgt)).sum().item()
        cos_src = (e(img) * e(src)).sum().item()
        tau = THRESHOLDS[name][1]  # FAR = 0.01
        rows.append((name, cos_tgt, cos_src, tau, cos_tgt > tau))
    print(f"\nPSNR vs source: {psnr(img, src).item():.2f} dB   SSIM vs source: {ssim_approx(img, src).item():.3f}")
    print(f"{'FR model':<11}{'cos(adv,tgt)':>14}{'cos(adv,src)':>14}{'tau@FAR1e-2':>13}  impersonation")
    for name, ct, cs, tau, ok in rows:
        print(f"{name:<11}{ct:>14.4f}{cs:>14.4f}{tau:>13.4f}  {'SUCCESS' if ok else 'fail'}")


if __name__ == "__main__":
    main()
