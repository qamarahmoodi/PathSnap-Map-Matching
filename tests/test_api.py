"""Tests for PathSnap FastAPI backend (Laptop 1): /health, /trips, /predict, and ground-truth isolation."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from pathsnap.api.app import create_app


def test_api_health_and_trips_endpoints():
    app = create_app(eval_partition="student_test")
    client = TestClient(app)

    # 1. GET /health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert h_data["status"] == "ok"
    assert h_data["ready"] is True
    assert h_data["model_loaded"] is True
    assert h_data["data_root_available"] is True
    assert h_data["permitted_trip_count"] == 10

    # 2. GET /trips
    r_trips = client.get("/trips")
    assert r_trips.status_code == 200
    t_data = r_trips.json()
    assert t_data["status"] == "ok"
    assert t_data["count"] == 10
    trip_ids = [t["trip_id"] for t in t_data["trips"]]
    assert "00000007" in trip_ids
    # Ensure no reference route data is exposed in /trips
    for item in t_data["trips"]:
        assert set(item.keys()) == {"trip_id", "display_label", "partition"}


def test_api_predict_success_and_strict_ground_truth_isolation():
    app = create_app(eval_partition="student_test")
    client = TestClient(app)

    r_pred = client.post("/predict", json={"trip_id": "00000007", "include_observation_records": True})
    assert r_pred.status_code == 200
    payload = r_pred.json()

    assert payload["status"] == "ok"
    assert payload["trip_id"] == "00000007"
    assert len(payload["predicted_route_arc_ids"]) > 0
    assert len(payload["predicted_route_segments_latlon"]) == len(payload["predicted_route_arc_ids"])
    assert len(payload["gps_trajectory_latlon"]) == payload["candidate_summary"]["num_observations"]
    assert len(payload["observation_predictions"]) == payload["candidate_summary"]["num_observations"]
    assert payload["processing_time_ms"]["total_request_ms"] > 0.0

    # Verify strict ground-truth isolation: response must NEVER contain reference route or ground-truth labels
    forbidden_keys = {
        "route",
        "reference_route",
        "reference_arc_id",
        "reference_route_position",
        "label",
        "ground_truth",
    }
    assert forbidden_keys.isdisjoint(set(payload.keys()))
    for obs_rec in payload["observation_predictions"]:
        assert forbidden_keys.isdisjoint(set(obs_rec.keys()))


def test_api_error_handling_invalid_trip_and_missing_model(tmp_path: Path):
    app = create_app(eval_partition="student_test")
    client = TestClient(app)

    # 1. Invalid / unpermitted trip_id -> 404
    r_bad = client.post("/predict", json={"trip_id": "99999999"})
    assert r_bad.status_code == 404

    # 2. Missing model artifact -> /health reports degraded (ready=False) and /predict returns 503
    app_no_model = create_app(
        eval_partition="student_test",
        model_path=tmp_path / "nonexistent_model.joblib",
    )
    client_no_model = TestClient(app_no_model)
    r_h = client_no_model.get("/health")
    assert r_h.status_code == 200
    assert r_h.json()["ready"] is False
    assert r_h.json()["model_loaded"] is False

    r_p = client_no_model.post("/predict", json={"trip_id": "00000007"})
    assert r_p.status_code == 503
