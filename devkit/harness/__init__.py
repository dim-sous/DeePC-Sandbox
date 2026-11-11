"""The registry of pieces. A name here is a package beside this file holding ``Config``, ``run``
and ``Record`` (``tests/test_devkit_layers.py`` holds that). Reading the registry imports
nothing; naming a piece imports that piece only (BESS's ``devkit/harness/__init__.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import ModuleType
from typing import Iterator

__all__ = ["NAMES", "COMPONENTS", "get_component"]

#: The registered pieces, in the order every surface shows them.
NAMES: tuple[str, ...] = ("closed_loop", "dataset", "system", "open_loop")


class _Registry(Mapping):
    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names
        self._loaded: dict[str, ModuleType] = {}

    def __getitem__(self, name: str) -> ModuleType:
        if name not in self._loaded:
            if name not in self._names:
                raise KeyError(name)
            self._loaded[name] = import_module(f".{name}", __name__)
        return self._loaded[name]

    def __contains__(self, name: object) -> bool:
        return name in self._names          # Mapping's default would import to answer

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


COMPONENTS: Mapping[str, ModuleType] = _Registry(NAMES)


def get_component(name: str) -> ModuleType:
    """The piece's module, imported on first use; an unknown name is refused with the list."""
    if name not in COMPONENTS:
        raise KeyError(f"unknown piece {name!r}; registered: {list(NAMES)}")
    return COMPONENTS[name]
