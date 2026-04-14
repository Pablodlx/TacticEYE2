"""Tests de integración de los endpoints REST de TacticEYE2."""
import io
import pytest
from tests.conftest import make_video_bytes


# ---------------------------------------------------------------------------
# /api/upload
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    def test_upload_valid_mp4_returns_session_id(self, client):
        data = make_video_bytes(4096)
        response = client.post(
            "/api/upload",
            files={"file": ("match.mp4", io.BytesIO(data), "video/mp4")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert "session_id" in body
        assert len(body["session_id"]) == 36  # UUID4

    def test_upload_rejects_unsupported_extension(self, client):
        response = client.post(
            "/api/upload",
            files={"file": ("script.sh", io.BytesIO(b"rm -rf /"), "text/plain")},
        )
        assert response.status_code == 400
        assert response.json()["success"] is False
        assert "Formato no soportado" in response.json()["error"]

    def test_upload_rejects_exe_extension(self, client):
        response = client.post(
            "/api/upload",
            files={"file": ("malware.exe", io.BytesIO(b"\x4d\x5a"), "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_upload_no_file_returns_422(self, client):
        response = client.post("/api/upload")
        assert response.status_code == 422

    def test_upload_allowed_extensions(self, client):
        for ext in [".avi", ".mov", ".mkv", ".webm"]:
            resp = client.post(
                "/api/upload",
                files={"file": (f"video{ext}", io.BytesIO(make_video_bytes(512)), "video/mp4")},
            )
            assert resp.status_code == 200, f"Extensión {ext} debería ser aceptada"


# ---------------------------------------------------------------------------
# /api/analyze/{session_id}
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    def test_analyze_unknown_session_returns_404(self, client):
        response = client.post("/api/analyze/nonexistent-session-id")
        assert response.status_code == 404
        assert response.json()["success"] is False

    def test_analyze_after_upload_starts_background(self, client):
        # Primero subimos un vídeo
        data = make_video_bytes(2048)
        upload_resp = client.post(
            "/api/upload",
            files={"file": ("game.mp4", io.BytesIO(data), "video/mp4")},
        )
        assert upload_resp.status_code == 200
        session_id = upload_resp.json()["session_id"]

        # Luego iniciamos el análisis
        analyze_resp = client.post(f"/api/analyze/{session_id}")
        # Debe devolver 200 o 202 (lanzado en background thread)
        assert analyze_resp.status_code in (200, 202)


# ---------------------------------------------------------------------------
# /api/status/{session_id}
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_status_unknown_session_returns_404(self, client):
        response = client.get("/api/status/does-not-exist")
        assert response.status_code == 404

    def test_status_after_upload_returns_uploaded(self, client):
        data = make_video_bytes(1024)
        upload_resp = client.post(
            "/api/upload",
            files={"file": ("clip.mp4", io.BytesIO(data), "video/mp4")},
        )
        session_id = upload_resp.json()["session_id"]

        status_resp = client.get(f"/api/status/{session_id}")
        assert status_resp.status_code == 200
        body = status_resp.json()
        # El estado puede estar en la raíz o anidado en 'data'
        status = body.get("status") or body.get("data", {}).get("status")
        assert status in ("uploaded", "processing", "completed", "error")


# ---------------------------------------------------------------------------
# / (index)
# ---------------------------------------------------------------------------


class TestIndexEndpoint:
    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
