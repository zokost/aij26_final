"""Explicit NC17 student defaults with validated research overrides."""
import json
import math


# This is the final student's recipe. Research variants opt in through JSON;
# adding an experimental augmentation or loss must not silently change NC17.
DEFAULT = dict(encoder="mit_b2", decoder_channels=320, img_size=640, batch=18,
               lr=1e-4, encoder_lr_mult=.1, schedule_epochs=40, final_epoch=32,
               seed=42, tversky_beta=.7, positive_balance=True,
               augmentation="minimal", initialization="checkpoint", stage2_kd_weight=.6)


def resolve(overrides=None):
    """Reject unknown keys and incompatible settings before allocating a GPU."""
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULT):
        raise ValueError("Unknown recipe keys or non-object recipe")
    cfg = {**DEFAULT, **overrides}
    if cfg["encoder"] not in ("mit_b1", "mit_b2"):
        raise ValueError("Supported student encoders: mit_b1, mit_b2")
    if cfg["augmentation"] not in ("minimal", "forensic"):
        raise ValueError("Supported augmentations: minimal, forensic")
    if cfg["initialization"] not in ("checkpoint", "imagenet"):
        raise ValueError("Supported initialization: checkpoint, imagenet")
    for key in ("decoder_channels", "img_size", "batch", "schedule_epochs", "final_epoch", "seed"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid integer: {key}")
    if cfg["img_size"] % 32 or cfg["final_epoch"] > cfg["schedule_epochs"]:
        raise ValueError("Input size must be divisible by32; final epoch must fit the schedule")
    for key in ("lr", "encoder_lr_mult"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Expected a finite positive value: {key}")
    beta = cfg["tversky_beta"]
    if beta is not None and (type(beta) not in (int, float) or not math.isfinite(beta) or not 0 < beta < 1):
        raise ValueError("Tversky beta must be in (0,1), or null for Dice")
    if type(cfg["positive_balance"]) is not bool:
        raise ValueError("positive_balance must be boolean")
    weight=cfg['stage2_kd_weight']
    if type(weight) not in (int,float) or not math.isfinite(weight) or weight<0:
        raise ValueError('stage2_kd_weight must be finite and nonnegative')
    return cfg


def load(path):
    return resolve(json.loads(path.read_text()) if path else None)
