"""Tests unitarios del sistema de alertas tácticas."""
import pytest

# ---------------------------------------------------------------------------
# MatchAlertSystem — imports ligeros, no necesita mock_modules
# possession_stats esperado: {'frames_by_team': {0: int, 1: int}, ...}
# ---------------------------------------------------------------------------

# frame_id que supera el check_interval por defecto (1800 frames @ 30fps = 60s)
CHECK_FRAME = 3600

# Posesión dominante: equipo 0 tiene 70%, equipo 1 tiene 30%
POSSESSION_DOM = {
    "frames_by_team": {0: 700, 1: 300},
    "passes_by_team": {0: 10, 1: 5},
    "current_team": 0,
    "possession_changes": 5,
}
# Posesión equilibrada: 52/48
POSSESSION_EQ = {
    "frames_by_team": {0: 520, 1: 480},
    "passes_by_team": {0: 8, 1: 7},
    "current_team": 0,
    "possession_changes": 5,
}


@pytest.fixture
def alert_system():
    from modules.match_alert_system import MatchAlertSystem
    return MatchAlertSystem()


class TestAlertIntervalGuard:
    def test_no_alert_before_check_interval(self, alert_system):
        """Con frame_id pequeño el sistema no debe generar ninguna alerta."""
        alerts = alert_system.analyze_and_generate_alerts(
            frame_id=10,
            possession_stats=POSSESSION_DOM,
            spatial_stats={},
        )
        assert alerts == [], "Demasiado pronto para disparar alertas"

    def test_possession_dominance_alert_fires_when_overdue(self, alert_system):
        """Debe disparar alerta de dominancia con posesión >= 60% y frame oportuno."""
        # El intervalo por defecto es 1800 frames; last_check_frame=0 (inicial)
        # y frame_id=CHECK_FRAME supera el umbral.
        alert_system.last_check_frame = 0

        alerts = alert_system.analyze_and_generate_alerts(
            frame_id=CHECK_FRAME,
            possession_stats=POSSESSION_DOM,
            spatial_stats={},
        )
        types = [a.type for a in alerts]
        # _check_possession_dominance emite alertas de tipo "possession"
        assert "possession" in types

    def test_no_possession_alert_below_threshold(self, alert_system):
        """Con posesión equilibrada no debe haber alerta de dominancia."""
        alert_system.last_check_frame = 0

        alerts = alert_system.analyze_and_generate_alerts(
            frame_id=CHECK_FRAME,
            possession_stats=POSSESSION_EQ,
            spatial_stats={},
        )
        types = [a.type for a in alerts]
        assert "possession" not in types


class TestAlertStructure:
    def test_alert_has_required_fields(self, alert_system):
        """Cada alerta debe tener type, message y severity."""
        alert_system.last_check_frame = 0

        alerts = alert_system.analyze_and_generate_alerts(
            frame_id=CHECK_FRAME,
            possession_stats=POSSESSION_DOM,
            spatial_stats={},
        )
        for alert in alerts:
            assert hasattr(alert, "type"), "Falta campo 'type'"
            assert hasattr(alert, "message"), "Falta campo 'message'"
            assert hasattr(alert, "severity"), "Falta campo 'severity'"

    def test_severity_values_are_valid(self, alert_system):
        """severity debe ser 'info', 'warning' o 'critical'."""
        VALID = {"info", "warning", "critical"}
        alert_system.last_check_frame = 0

        alerts = alert_system.analyze_and_generate_alerts(
            frame_id=CHECK_FRAME,
            possession_stats=POSSESSION_DOM,
            spatial_stats={},
        )
        for alert in alerts:
            assert alert.severity in VALID, f"Severity invalida: {alert.severity}"


class TestMinAlertInterval:
    def test_does_not_repeat_alerts_before_interval_elapses(self, alert_system):
        """La segunda llamada inmediata debe devolver [] porque no paso el intervalo."""
        alert_system.last_check_frame = 0

        alert_system.analyze_and_generate_alerts(
            frame_id=CHECK_FRAME, possession_stats=POSSESSION_DOM, spatial_stats={}
        )
        # last_check_frame se actualiza a CHECK_FRAME; avanzar solo 1 frame no alcanza el umbral
        second = alert_system.analyze_and_generate_alerts(
            frame_id=CHECK_FRAME + 1, possession_stats=POSSESSION_DOM, spatial_stats={}
        )
        assert second == [], "Segunda llamada inmediata no deberia generar alertas"
