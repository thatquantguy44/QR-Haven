from qr_haven import QRHavenSettings, ResearchPipeline
from qr_haven.integrations.market_terminal import TerminalPanel


def test_core_package_imports() -> None:
    settings = QRHavenSettings()
    pipeline = ResearchPipeline(steps=[])
    panel = TerminalPanel(panel_id="risk.summary", title="Risk Summary", kind="table")

    assert str(settings.data_dir) == "data"
    assert pipeline.steps == []
    assert panel.panel_id == "risk.summary"

