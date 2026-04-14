"""Fixtures compartidos para los tests de TacticEYE2."""
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def mock_modules():
    """Parchea las importaciones pesadas (YOLO, torch, cv2) para tests rápidos."""
    patches = [
        patch("ultralytics.YOLO", MagicMock()),
        patch("modules.reid_tracker.ReIDTracker", MagicMock()),
        patch("modules.possession_tracker_v2.PossessionTrackerV2", MagicMock()),
        patch("modules.team_classifier_v2.TeamClassifierV2", MagicMock()),
        patch("modules.video_sources.open_source", MagicMock()),
        patch("modules.match_analyzer.run_match_analysis", MagicMock()),
        patch("modules.match_state.FileSystemStorage", MagicMock()),
        patch("modules.field_heatmap_system.HeatmapAccumulator", MagicMock()),
        patch(
            "modules.field_heatmap_system.estimate_homography_with_flip_resolution",
            MagicMock(return_value=None),
        ),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


@pytest.fixture(scope="session")
def client(mock_modules):
    """Cliente de test con módulos pesados mockeados."""
    import app as application
    return TestClient(application.app)


def make_video_bytes(size: int = 1024) -> bytes:
    """Bytes sintéticos que simulan un pequeño archivo MP4."""
    # ftyp box mínimo para parecer un MP4 real
    return b"\x00\x00\x00\x1cftypisom\x00\x00\x00\x00isomavc1" + b"\x00" * max(0, size - 32)
