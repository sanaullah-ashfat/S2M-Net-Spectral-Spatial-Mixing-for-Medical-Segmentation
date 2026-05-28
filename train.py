#!/usr/bin/env python3
"""
S2M-Net Training Script
========================

Usage::

    # Standard training from a config file
    python train.py --config configs/retinal.yaml

    # Override any config key (dot-notation)
    python train.py --config configs/polyp.yaml training.epochs=50 model.sstm_k=32

    # Run a specific ablation study
    python train.py --config configs/polyp.yaml --ablation 7

    # Multi-GPU training
    python train.py --config configs/retinal.yaml training.gpus=[0,1]
"""

import os
import sys
import time
import json
import argparse
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import yaml
import tensorflow as tf

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2mnet.models          import S2MNet
from s2mnet.losses          import MorphologyAwareLoss
from s2mnet.dataloaders     import PatchDataset, FullImageDataset
from s2mnet.utils.metrics   import dice_coefficient, iou_score, precision_metric, recall_metric
from experiments.ablation_configs import get_ablation_config, list_ablation_configs


# =============================================================================
# Helpers
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_gpus(gpu_ids: list) -> tuple:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[GPU] No GPUs found — running on CPU.")
        return tf.distribute.get_strategy(), 0

    selected = [gpus[i] for i in gpu_ids if i < len(gpus)]
    try:
        tf.config.set_visible_devices(selected, "GPU")
        for g in selected:
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass

    n = len(selected)
    strategy = tf.distribute.MirroredStrategy() if n > 1 else tf.distribute.get_strategy()
    print(f"[GPU] Using {n} GPU(s): {[g.name for g in selected]}")
    return strategy, n


def load_config(config_path: str, overrides: list[str]) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides (key=value or key.subkey=value)
    for override in overrides:
        if "=" not in override:
            continue
        key_str, val_str = override.split("=", 1)
        keys = key_str.strip().split(".")

        # Try to parse value as Python literal
        try:
            import ast
            val = ast.literal_eval(val_str.strip())
        except Exception:
            val = val_str.strip()

        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

    return cfg


def build_lr_schedule(cfg: dict, base_lr: float):
    """Return a LearningRateScheduler callback."""
    schedule_type = cfg["training"].get("lr_schedule", "cosine_warmup")
    total_epochs  = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 10)
    min_lr        = cfg["training"].get("min_lr", 1e-6)

    def cosine_warmup(epoch, _lr):
        if epoch < warmup_epochs:
            return float(base_lr * (epoch + 1) / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return float(min_lr + (base_lr - min_lr) * (1 + np.cos(np.pi * progress)) / 2)

    if schedule_type == "cosine_warmup":
        return tf.keras.callbacks.LearningRateScheduler(cosine_warmup, verbose=0)
    return None


def build_dataloader(cfg: dict, split: str):
    mode = cfg["dataloader"].get("mode", "full_image")
    data_dirs = {
        "train": cfg["data"]["train_dir"],
        "val":   cfg["data"]["val_dir"],
        "test":  cfg["data"]["test_dir"],
    }
    dl_cfg = cfg["dataloader"]
    augment = split == "train"

    if mode == "patch":
        return PatchDataset(
            data_dir          = data_dirs[split],
            patch_size        = dl_cfg.get("patch_size", 256),
            stride            = dl_cfg.get("patch_stride", 32),
            min_fg_ratio      = dl_cfg.get("min_fg_ratio", 0.005),
            batch_size        = cfg["training"]["batch_size"],
            patches_per_epoch = dl_cfg.get("patches_per_epoch", 4000),
            augment           = augment,
            use_clahe         = dl_cfg.get("use_clahe", False),
            use_fov_mask      = dl_cfg.get("use_fov_mask", False),
            fov_margin        = dl_cfg.get("fov_margin", 20),
        )
    else:
        return FullImageDataset(
            data_dir         = data_dirs[split],
            input_size       = cfg["model"]["input_size"],
            num_classes      = cfg["data"].get("num_classes", 1),
            batch_size       = cfg["training"]["batch_size"],
            augment          = augment,
            expansion_factor = dl_cfg.get("expansion_factor", 1) if augment else 1,
        )


def build_model(cfg: dict) -> tf.keras.Model:
    m = cfg["model"]
    return S2MNet(
        input_size         = m["input_size"],
        num_classes        = cfg["data"].get("num_classes", 1),
        filters            = tuple(m.get("filters", [24, 32, 64, 80, 128])),
        use_mrfse          = m.get("use_mrfse", True),
        mrfse_kernels      = tuple(m.get("mrfse_kernels", [3, 5, 7])),
        se_reduction       = m.get("se_reduction", 16),
        expand_ratio       = m.get("expand_ratio", 6),
        use_sstm           = m.get("use_sstm", True),
        sstm_k             = m.get("sstm_k", 32),
        sstm_ssm_dim       = m.get("sstm_ssm_dim", 16),
        sstm_stages        = tuple(m.get("sstm_stages", [True]*5)),
        sstm_use_spectral  = tuple(m.get("sstm_use_spectral", [True]*5)),
        sstm_use_ssm       = tuple(m.get("sstm_use_ssm", [False, False, True, True, True])),
        sstm_dropout       = m.get("sstm_dropout", 0.1),
        use_bfp            = m.get("use_bfp", True),
        bfp_routing        = m.get("bfp_routing", "soft"),
        dropout            = m.get("dropout", 0.1),
        l2_reg             = m.get("l2_reg", 1e-4),
        activation         = m.get("activation", "elu"),
    )


def build_loss(cfg: dict) -> MorphologyAwareLoss:
    lc = cfg.get("loss", {})
    return MorphologyAwareLoss(
        components            = tuple(lc.get("components",
                                             ["core", "boundary", "structure", "scale", "texture"])),
        learned_weights       = lc.get("learned_weights", True),
        morphology_modulation = lc.get("morphology_modulation", True),
        coefficients          = lc.get("coefficients", None),
    )


# =============================================================================
# Main training loop
# =============================================================================

def train(cfg: dict) -> dict:
    set_seed(cfg["training"].get("seed", 42))

    save_dir = cfg["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # ── GPU ──────────────────────────────────────────────────────────────────
    gpu_ids  = cfg["training"].get("gpus", [0])
    strategy, n_gpu = setup_gpus(gpu_ids)

    # ── Dataloaders ──────────────────────────────────────────────────────────
    print("\n[Data] Building data loaders...")
    train_ds = build_dataloader(cfg, "train")
    val_ds   = build_dataloader(cfg, "val")
    print(f"  train steps/epoch: {len(train_ds)}")
    print(f"  val   steps/epoch: {len(val_ds)}")

    # ── Model + Loss ─────────────────────────────────────────────────────────
    base_lr = cfg["training"]["learning_rate"]

    with strategy.scope():
        model    = build_model(cfg)
        loss_fn  = build_loss(cfg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr, clipnorm=1.0)

        model.compile(
            optimizer = optimizer,
            loss      = loss_fn,
            metrics   = [dice_coefficient, iou_score, precision_metric, recall_metric],
        )

    model.summary(line_length=100)
    print(f"\n[Model] Total parameters: {model.count_params():,}")

    # ── Callbacks ────────────────────────────────────────────────────────────
    monitor      = cfg["training"].get("monitor", "val_dice_coefficient")
    monitor_mode = cfg["training"].get("monitor_mode", "max")
    patience     = cfg["training"].get("early_stopping_patience", 40)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath        = os.path.join(save_dir, "best_model.h5"),
            monitor         = monitor,
            mode            = monitor_mode,
            save_best_only  = True,
            verbose         = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = monitor,
            mode                 = monitor_mode,
            patience             = patience,
            verbose              = 1,
            restore_best_weights = True,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(save_dir, "training_log.csv")),
    ]

    lr_cb = build_lr_schedule(cfg, base_lr)
    if lr_cb is not None:
        callbacks.append(lr_cb)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[Train] Starting training for {cfg['training']['epochs']} epochs...")
    t0 = time.time()

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = cfg["training"]["epochs"],
        callbacks       = callbacks,
        verbose         = 1,
    )

    elapsed = time.time() - t0
    print(f"\n[Train] Finished in {elapsed/60:.1f} min")

    # ── Save learned MAL weights ──────────────────────────────────────────────
    if hasattr(loss_fn, "get_learned_weights"):
        learned_w = loss_fn.get_learned_weights()
        print(f"\n[Loss] Learned weights: {learned_w}")

    # ── Save config + metadata ────────────────────────────────────────────────
    metadata = {
        "training_time_minutes": elapsed / 60,
        "total_parameters":      int(model.count_params()),
        "config":                cfg,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Output] Files saved to: {save_dir}/")
    print(f"  ├─ best_model.h5")
    print(f"  ├─ training_log.csv")
    print(f"  └─ metadata.json")

    return {"history": history, "model": model}


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train S2M-Net")
    parser.add_argument("--config",  type=str, required=True, help="Path to YAML config")
    parser.add_argument("--ablation", type=int, default=None,
                        help="Ablation study ID (0–22). Overrides model + loss in config.")
    parser.add_argument("--list-ablations", action="store_true",
                        help="List all ablation configurations and exit.")
    parser.add_argument("overrides", nargs="*",
                        help="Additional overrides: key=value or section.key=value")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_ablations:
        list_ablation_configs()
        sys.exit(0)

    cfg = load_config(args.config, args.overrides)

    if args.ablation is not None:
        ab = get_ablation_config(args.ablation)
        print(f"\n[Ablation] #{args.ablation}: {ab['name']}")
        print(f"  {ab['description']}")

        # Merge ablation model + loss into config
        cfg["model"].update(ab["model"])
        cfg["loss"]  = ab["loss"]
        cfg["output"]["save_dir"] = os.path.join(
            cfg["output"].get("save_dir", "runs"),
            f"ablation_{args.ablation:02d}_{ab['name']}",
        )

    train(cfg)
