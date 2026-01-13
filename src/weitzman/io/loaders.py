"""
"""
import yaml
import numpy as np
from numpy.typing import NDArray
from typing import TypedDict

class Config(TypedDict):
    ...

def load_config_file(file_path: str) -> dict:

    with open(file_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        f.close()

    if raw is None:
        raise ValueError("YAML file is empty")

    if not isinstance(raw, dict):
        raise TypeError("YAML root must be a mapping, got {type(raw).__name__}")

    return raw


def load_labeled_points(file_path: str) -> NDArray[np.float64]:

    labels: list[str] = []
    array_points: list[list[float]] = []

    with open(file_path, "r") as f:
        for line in f:

            points, label = line.split(" -> ")

            labels.append(label.strip())
            array_points.append(list(map(float, points.split(" "))))
        f.close()

    data = np.array(array_points)
    return data


def load_lattice(file_path: str, labeled: bool = False) -> tuple[NDArray[np.float64], dict[int, NDArray[np.float64]]]:

    extension = file_path.rsplit(".", 1)[1]

    if labeled:
        data = load_labeled_points(file_path)

    elif (extension in ["pof", "POF"]) or (extension == "txt"):
        data = np.loadtxt(file_path)

    elif extension == "npy":
        data = np.load(file_path)
    
    else:
        raise ValueError("Lattice file extension must be one of the following: [.pof, .npy, .txt].")

    element_mapping = {i: e for i, e in enumerate(data)}
    return data, element_mapping
