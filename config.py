import os, json
import torch
from dataclasses import dataclass, field
from typing import Tuple, Dict

# --- Global Thresholds ---
THRESHOLDS = {
    'IR152': (0.094632, 0.166788, 0.227922),
    'IRSE50': (0.144840, 0.241045, 0.312703),
    'FaceNet': (0.256587, 0.409131, 0.591191),
    'MobileFace': (0.183635, 0.301611, 0.380878),
}

# Optional override: thresholds calibrated on impostor pairs for THIS pipeline (see scripts/calibrate_thresholds.py).
# The defaults above only fit FaceNet; ready-made files are in configs/.
if os.environ.get("ADV_THRESH_JSON"):
    with open(os.environ["ADV_THRESH_JSON"]) as f:
        THRESHOLDS = {k: tuple(v) for k, v in json.load(f).items()}

# Input resolution each FR model expects (square, pixels).
FR_INPUT_SIZE = {'IR152': 112, 'IRSE50': 112, 'MobileFace': 112, 'ArcFace': 112, 'AdaFace': 112, 'FaceNet': 160}

FACENET_TAU_1E3 = THRESHOLDS['IR152'][2]
SUCCESS_FAR = 0.01
_FAR_TO_INDEX = {0.1: 0, 0.01: 1, 0.001: 2}
SUCCESS_FAR_INDEX = _FAR_TO_INDEX.get(SUCCESS_FAR, 1)


@dataclass
class AdvTGDConfig:
    # --- Model & Env ---
    model_id: str = "sd2-community/stable-diffusion-2-1"
    resolution: int = 768
    # Using field(default_factory=...) to prevent issues with torch init at import time
    device: torch.device = field(default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # --- FR Models ---
    # env overrides (defaults unchanged): ADV_TRAIN_FRS="FaceNet,IRSE50,MobileFace" ADV_EVAL_FRS=IR152 ADV_K=3
    train_ensemble_frs: Tuple[str, ...] = field(default_factory=lambda: tuple(os.environ.get("ADV_TRAIN_FRS", "FaceNet,IRSE50").split(",")))
    eval_frs: Tuple[str, ...] = field(default_factory=lambda: tuple(os.environ.get("ADV_EVAL_FRS", "IR152").split(",")))
    ensemble_sample_k: int = int(os.environ.get("ADV_K", "2"))
    hinge_far_level: float = 0.01

    # --- Masking & Alignment (SGSM Default) ---
    align_faces: bool = True
    use_masking: bool = True
    use_hybrid_mask: bool = True  # SGSM Enabled by default (Method G)
    use_smart_masks: bool = False  # Pure Saliency
    use_parsing_mask: bool = False  # Pure Parsing
    use_heatmap_mask: bool = False
    heatmap_threshold: float = 0.30

    keep_percent_early: float = 0.12
    keep_percent_late: float = 0.35
    mask_expand_start_frac: float = 0.60
    feather_k: int = 7
    prior_strength: float = 0.5
    ellipse_rx_frac: float = 0.32
    ellipse_ry_frac: float = 0.40

    # --- Training Schedules (per pair) ---
    num_steps: int = 70
    lr: float = 5e-5
    decay_start_step: int = 30
    final_lr: float = 1e-5
    strength_train: float = 0.40
    use_fixed_t: bool = True
    t_fixed_frac: float = 0.6

    log_every: int = 5
    save_every: int = 1
    save_lora: bool = False

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 128
    lora_targets: Tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    lora_dropout: float = 0.08

    # --- Loss Weights (Main Phase Targets) ---
    lambda_eps: float = 1.5
    lambda_id_hinge: float = 2.5
    lambda_dir_warm: float = 0.6
    lambda_dir_main: float = 0.15
    lambda_src: float = 0.10
    src_margin: float = 0.10
    lambda_neg: float = 0.0
    neg_margin: float = 0.15
    lambda_cons: float = 0.10
    lambda_lpips: float = 0.02
    lambda_tv: float = 0.0
    lambda_bg_mse: float = 0.05

    w_text_early: float = 0.00
    w_text_late: float = 0.10
    text_late_start_frac: float = 0.60
    use_text_every_early: bool = False

    # --- Margins & Hinge Schedules ---
    base_margin_warm: float = 0.09
    base_margin_main: float = 0.20
    alpha_margin: float = 0.65
    blend_comp: float = 0.10
    margin_cap_warm: float = 0.28
    margin_cap_main: float = 0.65

    # --- Stall Controller ---
    warmup_frac: float = 0.10
    progress_threshold: float = 0.03
    stall_up_factor: float = 2.0
    stall_down_src: float = 1.8
    stall_max_triggers: int = 1

    # --- LPIPS Settings ---
    lpips_every: int = 4
    lpips_side: int = 768

    # --- EOT / Input Diversity Settings ---
    eot_scale_min: float = 1.0
    eot_scale_max: float = 1.0
    eot_blur_sigma: float = 0.0
    eot_noise_std: float = 0.0
    eot_color_delta: float = 0.0

    # --- Re-evaluation ---
    min_eval_step: int = 10
    eval_stride: int = 1
    blending_mode: str = 'alpha'
