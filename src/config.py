import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

def path(key: str) -> Path:
    return ROOT / CFG["paths"][key]