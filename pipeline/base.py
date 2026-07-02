"""Base class and registry for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class FeatureMeta:
    name: str
    description: str
    source: str  # e.g. "game_log", "statcast", "context"


class FeatureExtractor(ABC):
    """A single logical feature group.

    Subclasses implement ``extract()`` which returns a list of dicts.
    Each dict must contain ``player_id`` and ``game_pk`` as keys so the
    assembler can join across extractors.
    """

    @property
    @abstractmethod
    def features(self) -> list[FeatureMeta]:
        ...

    @abstractmethod
    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, type[FeatureExtractor]] = {}


def register(cls: type[FeatureExtractor]) -> type[FeatureExtractor]:
    _registry[cls.__name__] = cls
    return cls


def get_registry() -> dict[str, type[FeatureExtractor]]:
    return dict(_registry)
