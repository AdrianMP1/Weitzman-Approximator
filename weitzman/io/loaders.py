import yaml
import numpy as np

from numpy.typing import NDArray

def load_config_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"YAML file is empty: {file_path}")
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(raw).__name__}")

    return raw


def load_lattice(
    file_path: str, labeled: bool = False
) -> tuple[NDArray[np.float64], dict[int, str | NDArray[np.float64]]]:
    """
    Load a set of points from a .POF, .txt, or .npy file.

    Labeled format: each line is "<x> <y> <z> -> <label>"
      - element_mapping: {index -> label string}
    Unlabeled format:
      - element_mapping: {index -> coordinate array}

    Returns (data array of shape (n, d), element_mapping).
    """
    if labeled:
        data, labels = _load_labeled(file_path)
        element_mapping = {i: label for i, label in enumerate(labels)}
    else:
        ext = file_path.rsplit(".", 1)[-1].lower()
        if ext in ("pof", "txt"):
            data = np.loadtxt(file_path)
        elif ext == "npy":
            data = np.load(file_path)
        else:
            raise ValueError(f"Unsupported extension '.{ext}'. Use .pof, .txt, or .npy.")
        element_mapping = {i: row for i, row in enumerate(data)}

    return data, element_mapping


def _load_labeled(file_path: str) -> tuple[NDArray[np.float64], list[str]]:
    rows: list[list[float]] = []
    labels: list[str] = []
    with open(file_path, "r") as f:
        for line in f:
            points_str, label = line.split(" -> ")
            rows.append(list(map(float, points_str.split())))
            labels.append(label.strip())
    return np.array(rows), labels
