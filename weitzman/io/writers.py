import yaml
import numpy as np

from pathlib import Path
from numpy.typing import NDArray

def save_resolved_config(config: dict, run_dir: Path) -> None:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "resolved.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def write_values(data: NDArray, file_path: str) -> None:
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "npy":
        np.save(file_path, data)
    else:
        raise ValueError(f"Unsupported output format '.{ext}'. Use .npy.")
