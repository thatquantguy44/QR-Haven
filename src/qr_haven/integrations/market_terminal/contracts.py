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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this panel."""

        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "kind": self.kind,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TerminalPanel":
        """Reconstruct a panel from a plain mapping (e.g. parsed JSON)."""

        return cls(
            panel_id=str(data["panel_id"]),
            title=str(data["title"]),
            kind=str(data["kind"]),
            payload=dict(data.get("payload", {})),
        )


class TerminalPlugin(Protocol):
    """QR-Haven contribution that can be mounted by market_terminal."""

    plugin_id: str

    def panels(self) -> Sequence[TerminalPanel]:
        """Return terminal panels exposed by this plugin."""

