"""
"""
import yaml
import numpy as np
from numpy.typing import NDArray

from pathlib import Path

def save_resolved_config(config: dict, run_dir: Path) -> None:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / "resolved.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def write_pof(data: NDArray, file_path: str) -> None:
    """
    """
    with open(file_path, "w") as f:

        for row in data:
            numbers = ", ".join([e for e in row])
            line = "[" + numbers + "]\n"
            f.write(line)
        f.close()


def write_values(data: NDArray, file_path: str):

    extension = file_path.rsplit(".", 1)[1]
    if extension in ["pof", "POF"]:
        write_pof(data, file_path)

    elif extension == "npy":
        np.save(file_path, data)


def make_save_folder(folder_path: Path):
    folder_path.mkdir(parents=False, exist_ok=True)
