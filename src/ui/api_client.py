"""Configurable HTTP API client and deterministic Mock Adapter for Laptop 2 Evaluator UI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import urllib.error
import urllib.request


ENV_API_BASE_URL = "PATHSNAP_API_BASE_URL"
ENV_UI_MOCK_MODE = "PATHSNAP_UI_MOCK_MODE"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


# The 10 reserved evaluator-testing trip IDs (or development trip IDs returned by /trips)
MOCK_EVALUATOR_TRIPS: List[Dict[str, Any]] = [
    {"trip_id": "00000015", "partition": "evaluator_test", "num_observations": 1420, "label": "Trip 00000015 (1,420 GPS pts)"},
    {"trip_id": "00000017", "partition": "evaluator_test", "num_observations": 985, "label": "Trip 00000017 (985 GPS pts)"},
    {"trip_id": "00000023", "partition": "evaluator_test", "num_observations": 1610, "label": "Trip 00000023 (1,610 GPS pts)"},
    {"trip_id": "00000030", "partition": "evaluator_test", "num_observations": 1150, "label": "Trip 00000030 (1,150 GPS pts)"},
    {"trip_id": "00000046", "partition": "evaluator_test", "num_observations": 2105, "label": "Trip 00000046 (2,105 GPS pts)"},
    {"trip_id": "00000050", "partition": "evaluator_test", "num_observations": 890, "label": "Trip 00000050 (890 GPS pts)"},
    {"trip_id": "00000061", "partition": "evaluator_test", "num_observations": 1740, "label": "Trip 00000061 (1,740 GPS pts)"},
    {"trip_id": "00000074", "partition": "evaluator_test", "num_observations": 1320, "label": "Trip 00000074 (1,320 GPS pts)"},
    {"trip_id": "00000082", "partition": "evaluator_test", "num_observations": 1530, "label": "Trip 00000082 (1,530 GPS pts)"},
    {"trip_id": "00000085", "partition": "evaluator_test", "num_observations": 1080, "label": "Trip 00000085 (1,080 GPS pts)"},
]


@dataclass
class HealthResponse:
    ok: bool
    status: str
    model_loaded: bool
    model_name: str
    is_mock: bool
    api_base_url: str
    latency_ms: float
    active_partition: Optional[str] = None
    permitted_trip_count: int = 0
    error_message: Optional[str] = None


@dataclass
class TripsResponse:
    ok: bool
    trips: List[Dict[str, Any]]
    is_mock: bool
    partition: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class PredictResponse:
    """Normalized prediction response strictly excluding any ground-truth reference route or labels."""

    ok: bool
    trip_id: str
    is_mock: bool
    model_name: str
    decoder: str
    num_observations: int
    predicted_arc_ids: List[int]
    reconstructed_route_arcs: List[int]
    classifier_only_route_arcs: List[int] = field(default_factory=list)
    # Predicted route polyline segments: list of [[lat1, lon1], [lat2, lon2]]
    predicted_route_segments_latlon: List[List[List[float]]] = field(default_factory=list)
    classifier_only_route_segments_latlon: List[List[List[float]]] = field(default_factory=list)
    # Optional raw GPS points [[lat, lon], ...] for map overlay (never ground-truth route!)
    gps_points_latlon: List[List[float]] = field(default_factory=list)
    # Permitted server-calculated evaluation metrics (if returned by Laptop 1 API)
    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)
    decoder_summary: Dict[str, Any] = field(default_factory=dict)
    candidate_summary: Dict[str, Any] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    error_message: Optional[str] = None


def get_configured_api_base_url(override_url: Optional[str] = None) -> str:
    """Return the configured Laptop 1 FastAPI base URL without trailing slash."""
    url = (override_url or os.environ.get(ENV_API_BASE_URL) or DEFAULT_API_BASE_URL).strip()
    return url.rstrip("/")


def is_mock_mode_enabled(override_mock: Optional[bool] = None) -> bool:
    """Check whether explicit Mock Mode is enabled."""
    if override_mock is not None:
        return bool(override_mock)
    val = os.environ.get(ENV_UI_MOCK_MODE, "false").strip().lower()
    return val in ("1", "true", "yes", "mock")


def _build_mock_prediction(trip_id: str) -> PredictResponse:
    """Generate a clearly marked synthetic mock response for UI layout testing before Laptop 1 is live.

    NEVER loads any local model or reads any dataset file.
    """
    seed = sum(ord(c) for c in str(trip_id))
    base_lat = 49.845 + (seed % 7) * 0.002
    base_lon = 8.590 + (seed % 5) * 0.002

    segments = []
    gps_pts = []
    arc_ids = []
    for k in range(12):
        lat1 = base_lat + k * 0.0006
        lon1 = base_lon + k * 0.0008
        lat2 = base_lat + (k + 1) * 0.0006
        lon2 = base_lon + (k + 1) * 0.0008
        segments.append([[lat1, lon1], [lat2, lon2]])
        gps_pts.append([lat1 + 0.00008 * ((k % 3) - 1), lon1 - 0.00006 * ((k % 2) - 0.5)])
        arc_ids.append(1000 + k * 7)

    return PredictResponse(
        ok=True,
        trip_id=str(trip_id),
        is_mock=True,
        model_name="[MOCK MODE] Bayesian-Tuned HistGradientBoosting + HMM/Viterbi",
        decoder="Viterbi (Mock Contract Preview)",
        num_observations=len(gps_pts),
        predicted_arc_ids=arc_ids,
        reconstructed_route_arcs=arc_ids,
        predicted_route_segments_latlon=segments,
        gps_points_latlon=gps_pts,
        evaluation_metrics={
            "observation_top1_accuracy": 0.9142,
            "macro_f1": 0.8580,
            "route_continuity": 0.9825,
            "candidate_miss_rate": 0.0018,
            "note": "SYNTHETIC MOCK RESPONSE — NOT A LIVE EVALUATION RESULT",
        },
        inference_time_ms=42.5,
    )


class PathSnapAPIClient:
    """HTTP client for Laptop 2 Evaluator UI communicating with Laptop 1 FastAPI backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        mock_mode: Optional[bool] = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = get_configured_api_base_url(base_url)
        self.mock_mode = is_mock_mode_enabled(mock_mode)
        self.timeout_s = float(timeout_s)

    def _http_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw_body = resp.read().decode("utf-8")
            return json.loads(raw_body) if raw_body else {}

    def check_health(self) -> HealthResponse:
        """Call `GET /health` on Laptop 1 API."""
        t0 = time.perf_counter()
        if self.mock_mode:
            return HealthResponse(
                ok=True,
                status="mock_healthy",
                model_loaded=True,
                model_name="[MOCK MODE] Simulated Backend",
                is_mock=True,
                api_base_url=self.base_url,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        try:
            data = self._http_json("GET", "/health")
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            status = str(data.get("status", "ok"))
            ready = bool(data.get("ready", True))
            model_loaded = bool(data.get("model_loaded", True))
            return HealthResponse(
                ok=(status == "ok" and ready),
                status=status,
                model_loaded=model_loaded,
                model_name=str(data.get("model_name") or "PathSnap Server Model"),
                is_mock=False,
                api_base_url=self.base_url,
                latency_ms=elapsed_ms,
                active_partition=data.get("active_partition"),
                permitted_trip_count=int(data.get("permitted_trip_count", 0)),
                error_message=data.get("error"),
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return HealthResponse(
                ok=False,
                status="unreachable",
                model_loaded=False,
                model_name="Unknown",
                is_mock=False,
                api_base_url=self.base_url,
                latency_ms=elapsed_ms,
                error_message=f"Failed to connect to {self.base_url}/health: {exc}",
            )

    def get_trips(self) -> TripsResponse:
        """Call `GET /trips` on Laptop 1 API to retrieve the 10 evaluation trip options."""
        if self.mock_mode:
            return TripsResponse(ok=True, trips=list(MOCK_EVALUATOR_TRIPS), is_mock=True, partition="mock")

        try:
            data = self._http_json("GET", "/trips")
            raw_trips = data.get("trips", data) if isinstance(data, dict) else data
            part_name = data.get("partition") if isinstance(data, dict) else None
            normalized: List[Dict[str, Any]] = []
            if isinstance(raw_trips, list):
                for item in raw_trips:
                    if isinstance(item, str):
                        normalized.append({"trip_id": item, "label": f"Trip {item}"})
                    elif isinstance(item, dict) and "trip_id" in item:
                        tid = str(item["trip_id"])
                        obs_n = item.get("num_observations")
                        lbl = (
                            item.get("display_label")
                            or item.get("label")
                            or (f"Trip {tid} ({obs_n:,} GPS pts)" if obs_n else f"Trip {tid}")
                        )
                        normalized.append({**item, "trip_id": tid, "label": lbl})
            return TripsResponse(ok=True, trips=normalized, is_mock=False, partition=part_name)
        except Exception as exc:
            return TripsResponse(
                ok=False,
                trips=[],
                is_mock=False,
                error_message=f"Failed to fetch trips from {self.base_url}/trips: {exc}",
            )

    def predict_trip(self, trip_id: str) -> PredictResponse:
        """Call `POST /predict` with `{"trip_id": "<trip_id>"}` and sanitize response (never expose ground truth)."""
        tid = str(trip_id).strip()
        if not tid:
            return PredictResponse(
                ok=False,
                trip_id="",
                is_mock=self.mock_mode,
                model_name="",
                decoder="",
                num_observations=0,
                predicted_arc_ids=[],
                reconstructed_route_arcs=[],
                error_message="trip_id must not be empty.",
            )

        if self.mock_mode:
            return _build_mock_prediction(tid)

        t0 = time.perf_counter()
        try:
            data = self._http_json(
                "POST",
                "/predict",
                payload={"trip_id": tid, "include_observation_records": True},
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            # Strictly strip any accidental reference/ground-truth route fields if present
            safe_metrics = dict(data.get("evaluation_metrics") or data.get("metrics") or {})
            for forbidden_key in ("reference_route", "reference_arc_ids", "ground_truth", "labels", "route"):
                safe_metrics.pop(forbidden_key, None)

            # Normalize Qamar's real evaluation_metrics keys into UI display keys
            if "hmm_viterbi_top1_accuracy" in safe_metrics and "observation_top1_accuracy" not in safe_metrics:
                safe_metrics["observation_top1_accuracy"] = safe_metrics["hmm_viterbi_top1_accuracy"]
            if "candidate_macro_f1" in safe_metrics and "macro_f1" not in safe_metrics:
                safe_metrics["macro_f1"] = safe_metrics["candidate_macro_f1"]
            if "hmm_viterbi_route_continuity" in safe_metrics and "route_continuity" not in safe_metrics:
                safe_metrics["route_continuity"] = safe_metrics["hmm_viterbi_route_continuity"]

            obs_recs = data.get("observation_predictions") or []
            if obs_recs and "predicted_arc_id" in obs_recs[0]:
                pred_arcs = [int(r["predicted_arc_id"]) for r in obs_recs]
            else:
                pred_arcs = [int(x) for x in data.get("predicted_arc_ids", data.get("viterbi_arc_ids", []))]

            route_arcs = [
                int(x)
                for x in (
                    data.get("predicted_route_arc_ids")
                    or data.get("reconstructed_route_arcs")
                    or data.get("predicted_route")
                    or pred_arcs
                )
            ]
            cls_only_arcs = [int(x) for x in (data.get("classifier_only_route_arc_ids") or [])]

            segments = data.get("predicted_route_segments_latlon") or data.get("route_geometry_latlon") or []
            cls_only_segments = data.get("classifier_only_route_segments_latlon") or []
            gps_pts = (
                data.get("gps_trajectory_latlon")
                or data.get("gps_points_latlon")
                or data.get("track_latlon")
                or []
            )

            cand_sum = dict(data.get("candidate_summary") or {})
            dec_sum = dict(data.get("decoder_summary") or {})
            proc_ms = data.get("processing_time_ms") or {}
            num_obs = int(cand_sum.get("num_observations") or data.get("num_observations") or len(pred_arcs) or len(gps_pts))
            req_ms = float(
                proc_ms.get("total_request_ms") if isinstance(proc_ms, dict) and "total_request_ms" in proc_ms
                else data.get("inference_time_ms", elapsed_ms)
            )

            return PredictResponse(
                ok=True,
                trip_id=str(data.get("trip_id", tid)),
                is_mock=False,
                model_name=str(data.get("model_name", "Bayesian-Tuned HistGradientBoosting")),
                decoder=str(data.get("decoder", "HMM + Viterbi")),
                num_observations=num_obs,
                predicted_arc_ids=pred_arcs,
                reconstructed_route_arcs=route_arcs,
                classifier_only_route_arcs=cls_only_arcs,
                predicted_route_segments_latlon=segments,
                classifier_only_route_segments_latlon=cls_only_segments,
                gps_points_latlon=gps_pts,
                evaluation_metrics=safe_metrics,
                decoder_summary=dec_sum,
                candidate_summary=cand_sum,
                inference_time_ms=req_ms,
            )
        except urllib.error.HTTPError as hexc:
            err_body = ""
            try:
                err_body = hexc.read().decode("utf-8")
            except Exception:
                pass
            return PredictResponse(
                ok=False,
                trip_id=tid,
                is_mock=False,
                model_name="",
                decoder="",
                num_observations=0,
                predicted_arc_ids=[],
                reconstructed_route_arcs=[],
                error_message=f"HTTP {hexc.code} from {self.base_url}/predict: {err_body or hexc.reason}",
            )
        except Exception as exc:
            return PredictResponse(
                ok=False,
                trip_id=tid,
                is_mock=False,
                model_name="",
                decoder="",
                num_observations=0,
                predicted_arc_ids=[],
                reconstructed_route_arcs=[],
                error_message=f"Request to {self.base_url}/predict failed: {exc}",
            )
