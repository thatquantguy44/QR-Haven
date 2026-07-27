"""Integration contracts for the separate market_terminal project."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TerminalPanel:
    """Serializable description of one terminal panel."""

    panel_id: str
    title: str
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class TerminalPlugin(Protocol):
    """QR-Haven contribution that can be mounted by market_terminal."""

    plugin_id: str

    def panels(self) -> Sequence[TerminalPanel]:
        """Return terminal panels exposed by this plugin."""

