"""
Ablation Study Configurations
================================

23 pre-defined experiment configurations covering all ablation axes
from the S2M-Net paper.

Usage::

    python train.py --ablation 0    # Full model baseline
    python train.py --ablation 7    # No morphology modulation
    python train.py --ablation 15   # No SSTM

    # List all configurations
    python experiments/ablation_configs.py --list
"""

from __future__ import annotations

ABLATION_CONFIGS: dict[int, dict] = {

    # =====================================================================
    # 0  BASELINE — Full S2M-Net + Complete MAL
    # =====================================================================
    0: {
        "name": "FULL_MODEL",
        "description": "Full S2M-Net + Complete MAL (5 components + morphology)",
        "loss": {
            "components": ["core", "boundary", "structure", "scale", "texture"],
            "learned_weights": True,
            "morphology_modulation": True,
        },
        "model": {
            "use_mrfse": True, "use_sstm": True, "use_bfp": True,
            "sstm_stages": [True]*5, "sstm_k": 32,
            "bfp_routing": "soft",
        },
    },

    # =====================================================================
    # 1-5  MAL Component Ablations
    # =====================================================================
    1: {
        "name": "MAL_NO_CORE",
        "description": "MAL without Core Loss (L_core)",
        "loss": {
            "components": ["boundary", "structure", "scale", "texture"],
            "learned_weights": True, "morphology_modulation": True,
        },
        "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True,
                  "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"},
    },
    2: {
        "name": "MAL_NO_BOUNDARY",
        "description": "MAL without Boundary Loss (L_bnd)",
        "loss": {
            "components": ["core", "structure", "scale", "texture"],
            "learned_weights": True, "morphology_modulation": True,
        },
        "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True,
                  "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"},
    },
    3: {
        "name": "MAL_NO_STRUCTURE",
        "description": "MAL without Structure Loss (L_str)",
        "loss": {
            "components": ["core", "boundary", "scale", "texture"],
            "learned_weights": True, "morphology_modulation": True,
        },
        "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True,
                  "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"},
    },
    4: {
        "name": "MAL_NO_SCALE",
        "description": "MAL without Scale-Aware Focal Loss (L_sca)",
        "loss": {
            "components": ["core", "boundary", "structure", "texture"],
            "learned_weights": True, "morphology_modulation": True,
        },
        "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True,
                  "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"},
    },
    5: {
        "name": "MAL_NO_TEXTURE",
        "description": "MAL without Texture Loss (L_tex)",
        "loss": {
            "components": ["core", "boundary", "structure", "scale"],
            "learned_weights": True, "morphology_modulation": True,
        },
        "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True,
                  "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"},
    },

    # =====================================================================
    # 6-7  MAL Adaptation Mechanisms
    # =====================================================================
    6: {
        "name": "MAL_FIXED_WEIGHTS",
        "description": "Full MAL + morphology modulation, but fixed weights",
        "loss": {
            "components": ["core", "boundary", "structure", "scale", "texture"],
            "learned_weights": False, "morphology_modulation": True,
        },
        "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True,
                  "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"},
    },
    7: {
        "name": "MAL_NO_MORPHOLOGY",
        "description": "Full MAL + learned weights, but no morphology modulation (α_i=1)",
        "loss": {
            "components": ["core", "boundary", "structure", "scale", "texture"],
            "learned_weights": True, "morphology_modulation": False,
        },
        "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True,
                  "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"},
    },

    # =====================================================================
    # 8-11  SSTM Truncation Size K
    # =====================================================================
    8:  {"name": "SSTM_K16",  "description": "SSTM K=16",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [True]*5, "sstm_k": 16, "bfp_routing": "soft"}},
    9:  {"name": "SSTM_K24",  "description": "SSTM K=24",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [True]*5, "sstm_k": 24, "bfp_routing": "soft"}},
    10: {"name": "SSTM_K48",  "description": "SSTM K=48",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [True]*5, "sstm_k": 48, "bfp_routing": "soft"}},
    11: {"name": "SSTM_K64",  "description": "SSTM K=64",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [True]*5, "sstm_k": 64, "bfp_routing": "soft"}},

    # =====================================================================
    # 12-15  SSTM Stage Placement
    # =====================================================================
    12: {"name": "SSTM_EARLY_ONLY",  "description": "SSTM at stages 1-2 only",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [True, True, False, False, False], "sstm_k": 32, "bfp_routing": "soft"}},
    13: {"name": "SSTM_MIDDLE_ONLY", "description": "SSTM at stage 3 only",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [False, False, True, False, False], "sstm_k": 32, "bfp_routing": "soft"}},
    14: {"name": "SSTM_LATE_ONLY",   "description": "SSTM at stages 4-5 only",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [False, False, False, True, True], "sstm_k": 32, "bfp_routing": "soft"}},
    15: {"name": "NO_SSTM",          "description": "No SSTM at any stage",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": False, "use_bfp": True, "sstm_stages": [False]*5, "sstm_k": 32, "bfp_routing": "soft"}},

    # =====================================================================
    # 16-19  BFP Routing Variants
    # =====================================================================
    16: {"name": "BFP_HARD",    "description": "BFP with hard routing (threshold=0.5)",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "hard"}},
    17: {"name": "BFP_NO_ROUTING", "description": "BFP without routing (concatenate only)",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "none"}},
    18: {"name": "BFP_LEARNED", "description": "BFP with learned routing weights",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "learned"}},
    19: {"name": "NO_BFP",      "description": "Standard upsample+cat decoder (no BFP)",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": False, "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"}},

    # =====================================================================
    # 20-22  Architecture Component Ablations
    # =====================================================================
    20: {"name": "NO_MRFSE",      "description": "No MRF-SE blocks",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": False, "use_sstm": True, "use_bfp": True, "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"}},
    21: {"name": "VANILLA_UNET",  "description": "Vanilla U-Net (no MRFSE, SSTM, or BFP)",
         "loss": {"components": ["core","boundary","structure","scale","texture"], "learned_weights": True, "morphology_modulation": True},
         "model": {"use_mrfse": False, "use_sstm": False, "use_bfp": False, "sstm_stages": [False]*5, "sstm_k": 32, "bfp_routing": "soft"}},
    22: {"name": "SIMPLE_DICE",   "description": "Full architecture + simple Dice loss",
         "loss": {
             "components": ["core"],
             "learned_weights": False,
             "morphology_modulation": False,
         },
         "model": {"use_mrfse": True, "use_sstm": True, "use_bfp": True, "sstm_stages": [True]*5, "sstm_k": 32, "bfp_routing": "soft"}},
}


def get_ablation_config(ablation_id: int) -> dict:
    """Return the config dict for a given ablation ID."""
    if ablation_id not in ABLATION_CONFIGS:
        raise ValueError(
            f"Invalid ablation_id={ablation_id}. "
            f"Valid range: 0–{max(ABLATION_CONFIGS)}."
        )
    return ABLATION_CONFIGS[ablation_id]


def list_ablation_configs() -> None:
    """Print a formatted table of all ablation configurations."""
    categories = {
        "Baseline":                     [0],
        "MAL Component Ablations":      [1, 2, 3, 4, 5],
        "MAL Adaptation Mechanisms":    [6, 7],
        "SSTM Truncation Size (K)":     [8, 9, 10, 11],
        "SSTM Stage Placement":         [12, 13, 14, 15],
        "BFP Routing Variants":         [16, 17, 18, 19],
        "Architecture Components":      [20, 21, 22],
    }
    print("\n" + "="*80)
    print("S2M-Net Ablation Study Configurations")
    print("="*80)
    for category, ids in categories.items():
        print(f"\n  {category}:")
        for i in ids:
            cfg = ABLATION_CONFIGS[i]
            print(f"    [{i:2d}] {cfg['name']:<25s} – {cfg['description']}")
    print("="*80)


if __name__ == "__main__":
    import sys
    if "--list" in sys.argv:
        list_ablation_configs()
