"""Declarative registries for ML-AMPSIT extension points.

New regressors and feature transforms are added by registering a specification;
the analysis core and GUI do not need to know their concrete classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Callable, Generic, Iterable, TypeVar


T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, key: str, item: T) -> T:
        normalized = key.strip().lower()
        if normalized in self._items:
            raise KeyError(f"Duplicate {self.kind} key: {normalized}")
        self._items[normalized] = item
        return item

    def get(self, key: str) -> T:
        normalized = str(key).strip().lower()
        try:
            return self._items[normalized]
        except KeyError as error:
            choices = ", ".join(self._items)
            raise ValueError(f"Unknown {self.kind} '{key}'. Available: {choices}") from error

    def keys(self) -> tuple[str, ...]:
        return tuple(self._items)

    def values(self) -> tuple[T, ...]:
        return tuple(self._items.values())

    def items(self) -> tuple[tuple[str, T], ...]:
        return tuple(self._items.items())

    def __contains__(self, key: object) -> bool:
        return str(key).strip().lower() in self._items


@dataclass(frozen=True)
class BuildContext:
    seed: int
    config: dict[str, Any] = field(default_factory=dict)
    n_features: int | None = None

    def options(self, section: str, key: str) -> dict[str, Any]:
        return dict(self.config.get(section, {}).get(key, {}))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    factory: Callable[[BuildContext], Any]
    search_space: Callable[[BuildContext], dict[str, Any]] = lambda _context: {}
    tuning_factory: Callable[[BuildContext], Any] | None = None
    default_importance: str = "pfi"
    tree_based: bool = False
    dependency: str | None = None
    requirements_file: str = "requirements-optional.txt"
    experimental: bool = False
    description: str = ""

    @property
    def available(self) -> bool:
        return self.dependency is None or find_spec(self.dependency) is not None

    def require_dependency(self) -> None:
        if not self.available:
            raise ModuleNotFoundError(
                f"Model '{self.key}' requires the optional package '{self.dependency}'. "
                f"Install {self.requirements_file} in the project environment."
            )


@dataclass(frozen=True)
class TransformSpec:
    key: str
    label: str
    factory: Callable[[BuildContext], Any]
    dependency: str | None = None
    requirements_file: str = "requirements-optional.txt"
    experimental: bool = False
    description: str = ""

    @property
    def available(self) -> bool:
        return self.dependency is None or find_spec(self.dependency) is not None

    def require_dependency(self) -> None:
        if not self.available:
            raise ModuleNotFoundError(
                f"Transform '{self.key}' requires the optional package '{self.dependency}'. "
                f"Install {self.requirements_file} in the project environment."
            )


@dataclass(frozen=True)
class ImportanceSpec:
    key: str
    label: str
    compute: Callable[..., Any]
    experimental: bool = False
    description: str = ""

    @property
    def available(self) -> bool:
        return True


MODEL_REGISTRY: ComponentRegistry[ModelSpec] = ComponentRegistry("model")
TRANSFORM_REGISTRY: ComponentRegistry[TransformSpec] = ComponentRegistry("transform")
IMPORTANCE_REGISTRY: ComponentRegistry[ImportanceSpec] = ComponentRegistry("importance method")


def display_choices(specs: Iterable[Any]) -> list[tuple[str, str]]:
    choices = []
    for spec in specs:
        qualifiers = []
        if getattr(spec, "experimental", False):
            qualifiers.append("experimental")
        if not getattr(spec, "available", True):
            qualifiers.append("optional dependency missing")
        suffix = f" ({', '.join(qualifiers)})" if qualifiers else ""
        choices.append((f"{spec.label}{suffix}", spec.key))
    return choices
