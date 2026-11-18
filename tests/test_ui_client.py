"""Tests for Laptop 2 Evaluator UI client, Mock Mode isolation, and live HTTP server mock adapter."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest

from ui.api_client import (
    MOCK_EVALUATOR_TRIPS,
    PathSnapAPIClient,
    get_configured_api_base_url,
    is_mock_mode_enabled,
)
from ui.app import build_prediction_folium_map_html, render_dashboard_page


def test_ui_mock_mode_and_zero_ground_truth_leakage():
    client = PathSnapAPIClient(base_url="http://192.168.1.50:8000", mock_mode=True)

    health = client.check_health()
    assert health.ok is True
    assert health.is_mock is True
    assert "MOCK" in health.model_name

    trips = client.get_trips()
    assert trips.ok is True
    assert trips.is_mock is True
    assert len(trips.trips) == 10
    assert trips.trips[0]["trip_id"] == MOCK_EVALUATOR_TRIPS[0]["trip_id"]

    pred = client.predict_trip("00000015")
    assert pred.ok is True
    assert pred.is_mock is True
    assert pred.trip_id == "00000015"
    assert len(pred.reconstructed_route_arcs) > 0
    assert len(pred.predicted_route_segments_latlon) > 0

    # Render Folium map and full HTML page
    map_html = build_prediction_folium_map_html(pred)
    assert "folium" in map_html.lower() or "leaflet" in map_html.lower()

    page_html = render_dashboard_page(
        api_base_url="http://192.168.1.50:8000",
        mock_mode=True,
        selected_trip_id="00000015",
        run_prediction=True,
    )
    assert "MOCK CONTRACT PREVIEW MODE" in page_html
    assert "00000015" in page_html


def test_ui_live_http_client_strips_forbidden_ground_truth_fields():
    """Spin up a lightweight local HTTP server simulating Laptop 1's FastAPI endpoints
    and verify that `PathSnapAPIClient` parses `/health`, `/trips`, and `/predict`
    and strips any forbidden ground-truth fields (`reference_route`, `ground_truth`, `labels`).
    """

    class DummyLaptop1Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = json.dumps({"status": "ok", "model_loaded": True, "model_name": "Tuned HGB"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/trips":
                body = json.dumps({"trips": ["00000000", "00000007"]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/predict":
                length = int(self.headers.get("Content-Length", "0"))
                req_data = json.loads(self.rfile.read(length).decode("utf-8"))
                tid = req_data["trip_id"]
                resp_payload = {
                    "trip_id": tid,
                    "model_name": "Bayesian-Tuned HistGradientBoostingClassifier",
                    "decoder": "HMM + Viterbi",
                    "num_observations": 3,
                    "predicted_arc_ids": [10, 10, 15],
                    "reconstructed_route_arcs": [10, 15],
                    "predicted_route_segments_latlon": [[[49.84, 8.59], [49.85, 8.60]]],
                    "evaluation_metrics": {
                        "observation_top1_accuracy": 0.92,
                        "macro_f1": 0.85,
                        "route_continuity": 1.0,
                        "reference_route": [10, 15],  # Forbidden field that must be stripped!
                        "ground_truth": [1, 1, 1],    # Forbidden field that must be stripped!
                    },
                    "inference_time_ms": 28.4,
                }
                body = json.dumps(resp_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), DummyLaptop1Handler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        client = PathSnapAPIClient(base_url=f"http://127.0.0.1:{port}", mock_mode=False, timeout_s=5.0)
        h = client.check_health()
        assert h.ok is True
        assert h.is_mock is False
        assert h.model_name == "Tuned HGB"

        t = client.get_trips()
        assert t.ok is True
        assert len(t.trips) == 2
        assert t.trips[0]["trip_id"] == "00000000"

        p = client.predict_trip("00000007")
        assert p.ok is True
        assert p.is_mock is False
        assert p.trip_id == "00000007"
        assert p.reconstructed_route_arcs == [10, 15]
        assert "reference_route" not in p.evaluation_metrics
        assert "ground_truth" not in p.evaluation_metrics
        assert p.evaluation_metrics["observation_top1_accuracy"] == 0.92
    finally:
        server.shutdown()
        server.server_close()


def test_ui_client_against_real_qamar_fastapi_backend():
    """End-to-end integration test running Qamar's real FastAPI server (`pathsnap.api.app`)
    in a background Uvicorn thread and verifying Pranav's `PathSnapAPIClient` + `render_dashboard_page`.
    """
    import socket
    import time
    import uvicorn
    from pathsnap.api.app import create_app

    # Find an open ephemeral TCP port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    fastapi_app = create_app(eval_partition="student_test")
    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        client = PathSnapAPIClient(base_url=base_url, mock_mode=False, timeout_s=30.0)

        # Wait up to 10s for uvicorn startup
        h = None
        for _ in range(40):
            h = client.check_health()
            if h.ok:
                break
            time.sleep(0.25)

        assert h is not None and h.ok is True
        assert h.model_loaded is True
        assert h.permitted_trip_count == 10

        trips = client.get_trips()
        assert trips.ok is True
        assert len(trips.trips) == 10
        assert any(t["trip_id"] == "00000007" for t in trips.trips)

        pred = client.predict_trip("00000007")
        assert pred.ok is True
        assert pred.is_mock is False
        assert pred.trip_id == "00000007"
        assert len(pred.reconstructed_route_arcs) > 0
        assert len(pred.predicted_route_segments_latlon) > 0
        assert len(pred.gps_points_latlon) == pred.num_observations
        assert "observation_top1_accuracy" in pred.evaluation_metrics
        assert "reference_route" not in pred.evaluation_metrics

        page_html = render_dashboard_page(
            api_base_url=base_url,
            mock_mode=False,
            selected_trip_id="00000007",
            run_prediction=True,
        )
        assert "LIVE API MODE (LAPTOP 1 FASTAPI)" in page_html
        assert "00000007" in page_html
        assert "ONLINE" in page_html
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
