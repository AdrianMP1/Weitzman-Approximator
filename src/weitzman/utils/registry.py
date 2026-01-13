import importlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")

@dataclass(frozen=True)
class LazyTarget:
    """
    Points to something importable in a lazy way.

    module: python module path, e.g., "weitzman.heuristics.greedy"
    attr: attribute within module, e.g., "nearest_neighbour" 
    kind: optional label for error messages
    """
    module: str
    attr: str
    kind: str = "component"

# Custom exception type
class RegistryError(RuntimeError):
    pass


def _import_attr(target: LazyTarget) -> Any:
    mod = importlib.import_module(target.module)
    obj = getattr(mod, target.attr, None)

    if obj is None:
        raise RegistryError(
            f"{target.kind} target not found: {target.module}.{target.attr}"
        )
    return obj


def resolve_callable(target: LazyTarget) -> Callable[..., Any]:
    obj = _import_attr(target)

    if not callable(obj):
        raise RegistryError(
            f"{target.kind} target is not callable: {target.module}.{target.attr}"
        )
    return obj


def resolve_class(target: LazyTarget) -> type:
    obj = _import_attr(target)
    if not isinstance(obj, type):
        raise RegistryError(
            f"{target.kind} target is not a class: {target.module}.{target.attr}"
        )
    return obj


def build_component(
    registry: dict[str, LazyTarget],
    name: str,
    *,
    builder: str = "callable"
) -> Any:
    """
    Generic factory
    """
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise RegistryError(f"Unknown component '{name}'. Available: {available}")

    target = registry[name]

    if builder == "callable":
        fn = resolve_callable(target)
        return fn

    raise ValueError(f"Unknown builder mode: {builder}")
