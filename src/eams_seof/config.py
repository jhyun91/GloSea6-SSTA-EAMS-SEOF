from __future__ import annotations

from pathlib import Path
import copy
import yaml


def _deep_update(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, repo_root: str | Path | None = None) -> dict:
    'Load a YAML model configuration.'
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "inherits" in cfg:
        inherit_path = Path(cfg.pop("inherits"))
        if not inherit_path.is_absolute() and repo_root is not None:
            inherit_path = Path(repo_root) / inherit_path
        with inherit_path.open("r", encoding="utf-8") as f:
            parent = yaml.safe_load(f)
        cfg = _deep_update(parent, cfg)

    return cfg


def resolve_path(repo_root: str | Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(repo_root) / p
