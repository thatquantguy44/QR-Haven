"""Registry for QR-Haven components that should appear in market_terminal."""

from qr_haven.integrations.market_terminal.contracts import TerminalPlugin


class TerminalPluginRegistry:
    """In-memory registry used until market_terminal owns plugin discovery."""

    def __init__(self) -> None:
        self._plugins: dict[str, TerminalPlugin] = {}

    def register(self, plugin: TerminalPlugin) -> None:
        """Register or replace a terminal plugin by id."""

        self._plugins[plugin.plugin_id] = plugin

    def get(self, plugin_id: str) -> TerminalPlugin:
        """Return a registered plugin."""

        return self._plugins[plugin_id]

    def all(self) -> list[TerminalPlugin]:
        """Return all registered plugins in deterministic order."""

        return [self._plugins[key] for key in sorted(self._plugins)]

