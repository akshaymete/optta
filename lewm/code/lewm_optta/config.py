"""Load config.yaml, apply task / arm / cell presets, then strict dotlist overrides (typos raise)."""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = Path(__file__).with_name("config.yaml")
PRESET_KEYS = ("task", "arm", "cell")      # applied in this order; explicit overrides win
TABLES = {"task": "tasks", "arm": "arms", "cell": "cells"}


def load_config(argv: list[str] | None = None) -> DictConfig:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = OmegaConf.load(CONFIG_PATH)

    presets, overrides = [], []
    for arg in argv:
        key, sep, val = arg.partition("=")
        if not sep:
            raise SystemExit(f"bad override {arg!r}; expected key=value")
        (presets if key in PRESET_KEYS else overrides).append((key, val))

    presets.sort(key=lambda kv: PRESET_KEYS.index(kv[0]))
    for key, val in presets:
        table = cfg[TABLES[key]]
        if val not in table:
            raise SystemExit(f"unknown {key} {val!r}; choose from {list(table.keys())}")
        preset = OmegaConf.create(OmegaConf.to_container(table[val], resolve=False) or {})
        cfg = OmegaConf.merge(cfg, preset)
        cfg[key] = val

    OmegaConf.set_struct(cfg, True)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides]))

    if cfg.run_name is None:
        task = "" if cfg.task in (None, "pusht") else f"{cfg.task}_"
        c = cfg.corruption
        cell = f"{c.name}{c.level}" if c.name else (cfg.cell or "clean")
        enc = "_enc" if cfg.adapt.finetune_encoder else ""
        cfg.run_name = f"{task}{cell}_{cfg.arm or 'custom'}{enc}_s{cfg.seed}{cfg.tag or ''}"
    OmegaConf.resolve(cfg)
    return cfg
