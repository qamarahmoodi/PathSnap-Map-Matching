"""FastAPI backend for Laptop 1 in PathSnap two-laptop deployment.

Owner: Qamar Ahmed (Technical Lead)

Endpoints:
  - GET  /health   : Reports API health, model loaded state, and dataset readiness.
  - GET  /trips    : Returns the 10 permitted evaluation trip options (safe IDs & labels only).
  - POST /predict  : Runs the complete route-free inference pipeline (`run_trip_inference`),
                     followed by isolated server-side evaluation (`evaluate_trip_predictions_server_side`),
                     logging every step visibly to the Laptop 1 console and never exposing ground-truth
                     `.route` sequences or labels to the client/UI.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathsnap.api.schemas import (
    HealthResponse,
    ObservationPredictionRecord,
    PredictRequest,
    PredictResponse,
    TripOption,
    TripsListResponse,
)
from pathsnap.data.loader import list_trips, resolve_data_root, resolve_partition_dir
from pathsnap.inference.inference_pipeline import run_trip_inference
from pathsnap.inference.model_scorer import CandidateModelScorer
from pathsnap.inference.server_evaluator import evaluate_trip_predictions_server_side
from pathsnap.schemas.candidate_schema import CandidateConfig
from sequence.viterbi_decoder import ViterbiConfig


# Configure visible console logger for Laptop 1 terminal demonstration
logger = logging.getLogger("pathsnap.api")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [LAPTOP-1 API] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


ENV_EVAL_PARTITION = "PATHSNAP_EVAL_PARTITION"
ENV_CORS_ORIGINS = "PATHSNAP_CORS_ORIGINS"


class BackendRuntimeState:
    """Holds singleton model scorer, resolved data root, and permitted trip allowlist."""

    def __init__(
        self,
        data_root: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
        eval_partition: Optional[str] = None,
        candidate_config: Optional[CandidateConfig] = None,
        viterbi_config: Optional[ViterbiConfig] = None,
    ) -> None:
        self.custom_data_root = data_root
        self.custom_model_path = model_path
        self.custom_eval_partition = eval_partition
        self.candidate_config = candidate_config or CandidateConfig()
        self.viterbi_config = viterbi_config or ViterbiConfig()

        self.scorer: Optional[CandidateModelScorer] = None
        self.model_error: Optional[str] = None
        self.data_root: Optional[Path] = None
        self.active_partition: Optional[str] = None
        self.permitted_trips: Dict[str, str] = {}  # trip_id -> partition
        self.refresh()

    def refresh(self) -> None:
        """Load or reload model artifact and discover permitted evaluation trips."""
        # 1. Load model artifact once
        if self.scorer is None:
            try:
                self.scorer = CandidateModelScorer.from_path(self.custom_model_path)
                self.model_error = None
                logger.info(
                    "Loaded model '%s' from %s",
                    self.scorer.model_name,
                    self.scorer.model_path,
                )
            except Exception as exc:
                self.scorer = None
                self.model_error = str(exc)
                logger.error("Failed to load model artifact: %s", exc)

        # 2. Resolve dataset root and permitted evaluation partition
        try:
            self.data_root = resolve_data_root(self.custom_data_root)
            part_pref = (
                self.custom_eval_partition
                or os.environ.get(ENV_EVAL_PARTITION)
            )
            chosen_part = "student_test"
            if part_pref:
                try:
                    resolve_partition_dir(part_pref, self.data_root)
                    chosen_part = part_pref
                except FileNotFoundError:
                    chosen_part = "student_test"
            else:
                # Automatically prefer evaluator_test if available in data_root or sibling ../evaluator_test
                try:
                    resolve_partition_dir("evaluator_test", self.data_root)
                    chosen_part = "evaluator_test"
                except FileNotFoundError:
                    chosen_part = "student_test"

            self.active_partition = chosen_part
            trip_ids = list_trips(chosen_part, self.data_root)
            self.permitted_trips = {tid: chosen_part for tid in trip_ids}
            logger.info(
                "Resolved data_root=%s | active_partition=%s | permitted_trips=%d (%s)",
                self.data_root,
                self.active_partition,
                len(trip_ids),
                ", ".join(trip_ids[:10]),
            )
        except Exception as exc:
            self.data_root = None
            self.active_partition = None
            self.permitted_trips = {}
            logger.error("Failed to resolve dataset root: %s", exc)


def create_app(
    data_root: Optional[str | Path] = None,
    model_path: Optional[str | Path] = None,
    eval_partition: Optional[str] = None,
    scorer: Optional[CandidateModelScorer] = None,
) -> FastAPI:
    """Create and configure the PathSnap FastAPI application."""
    app = FastAPI(
        title="PathSnap Map-Matching & Route Reconstruction API (Laptop 1)",
        version="2.0.0",
        description=(
            "Distributed GPS Map-Matching backend hosting spatial candidate generation, "
            "trained ML candidate scoring, HMM/Viterbi decoding, and isolated server-side evaluation."
        ),
    )

    cors_env = os.environ.get(ENV_CORS_ORIGINS, "*")
    origins = [o.strip() for o in cors_env.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    state = BackendRuntimeState(
        data_root=data_root,
        model_path=model_path,
        eval_partition=eval_partition,
    )
    if scorer is not None:
        state.scorer = scorer
        state.model_error = None

    app.state.runtime = state

    @app.get("/health", response_model=HealthResponse)
    def get_health() -> HealthResponse:
        rt: BackendRuntimeState = app.state.runtime
        model_ok = rt.scorer is not None
        data_ok = rt.data_root is not None and len(rt.permitted_trips) > 0
        ready = bool(model_ok and data_ok)
        return HealthResponse(
            status="ok" if ready else "degraded",
            ready=ready,
            model_loaded=model_ok,
            model_name=rt.scorer.model_name if rt.scorer is not None else None,
            model_path=rt.scorer.model_path if rt.scorer is not None else None,
            data_root_available=data_ok,
            active_partition=rt.active_partition,
            permitted_trip_count=len(rt.permitted_trips),
            error=rt.model_error if not model_ok else (None if data_ok else "Dataset root or trips unavailable"),
        )

    @app.get("/trips", response_model=TripsListResponse)
    def get_trips() -> TripsListResponse:
        rt: BackendRuntimeState = app.state.runtime
        if rt.data_root is None or not rt.permitted_trips:
            raise HTTPException(
                status_code=503,
                detail="Dataset root or evaluation trips are not available on the server.",
            )
        options = [
            TripOption(
                trip_id=tid,
                display_label=f"Evaluation Trip {idx + 1} ({tid})",
                partition=part,
            )
            for idx, (tid, part) in enumerate(sorted(rt.permitted_trips.items()))
        ]
        logger.info("GET /trips -> returned %d permitted trip options", len(options))
        return TripsListResponse(
            status="ok",
            partition=rt.active_partition or "unknown",
            count=len(options),
            trips=options,
        )

    @app.post("/predict", response_model=PredictResponse)
    def post_predict(req: PredictRequest) -> PredictResponse:
        t_req_start = time.perf_counter()
        rt: BackendRuntimeState = app.state.runtime
        trip_id = str(req.trip_id).strip()

        logger.info("============================================================")
        logger.info("POST /predict received | requested trip_id='%s'", trip_id)

        if rt.scorer is None:
            logger.error("POST /predict aborted: model is not loaded (%s)", rt.model_error)
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact is not loaded on server: {rt.model_error}",
            )

        if rt.data_root is None or not rt.permitted_trips:
            logger.error("POST /predict aborted: dataset root unavailable")
            raise HTTPException(
                status_code=503,
                detail="Dataset root is not configured or unavailable on server.",
            )

        if trip_id not in rt.permitted_trips:
            logger.warning(
                "POST /predict rejected: trip_id='%s' not in permitted allowlist %s",
                trip_id,
                list(rt.permitted_trips.keys()),
            )
            raise HTTPException(
                status_code=404,
                detail=f"Trip '{trip_id}' is not in the permitted evaluation trip list.",
            )

        partition = rt.permitted_trips[trip_id]
        try:
            logger.info("Starting route-free inference pipeline for trip_id='%s' (partition='%s')", trip_id, partition)
            inf_out = run_trip_inference(
                trip_id=trip_id,
                partition=partition,
                data_root=rt.data_root,
                scorer=rt.scorer,
                candidate_config=rt.candidate_config,
                viterbi_config=rt.viterbi_config,
            )
            logger.info(
                "Candidate generation & scoring complete | obs=%d | candidates=%d | model=%s | score_ms=%.2f",
                inf_out.candidate_summary["num_observations"],
                inf_out.candidate_summary["total_candidates"],
                inf_out.model_name,
                inf_out.timing_ms["model_scoring_ms"],
            )
            logger.info(
                "HMM/Viterbi decoding complete | route_arcs=%d | viterbi_continuity=%.4f | decode_ms=%.2f",
                len(inf_out.predicted_route_arc_ids),
                inf_out.decoder_summary.get("viterbi_route_continuity", 0.0),
                inf_out.timing_ms["viterbi_decoding_ms"],
            )

            # Separate internal server-side evaluation (never exposes .route to response)
            eval_res = evaluate_trip_predictions_server_side(
                inference_output=inf_out,
                data_root=rt.data_root,
                candidate_config=rt.candidate_config,
            )
            logger.info(
                "Server-side evaluation complete | ref_available=%s | top1_acc=%s | eval_ms=%.2f",
                eval_res.reference_available,
                eval_res.metrics.get("hmm_viterbi_top1_accuracy"),
                eval_res.evaluation_time_ms,
            )
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            logger.error("Missing trip files for '%s': %s", trip_id, exc)
            raise HTTPException(status_code=404, detail=f"Trip files missing for '{trip_id}': {exc}") from exc
        except Exception as exc:
            logger.exception("Inference pipeline failed for trip '%s'", trip_id)
            raise HTTPException(status_code=500, detail=f"Inference pipeline error: {exc}") from exc

        obs_records: List[ObservationPredictionRecord] = []
        if req.include_observation_records:
            df_p = inf_out.observation_predictions
            for row in df_p.itertuples(index=False):
                obs_records.append(
                    ObservationPredictionRecord(
                        observation_id=int(row.observation_id),
                        timestamp_s=float(row.timestamp_s),
                        predicted_arc_id=int(row.viterbi_arc_id),
                        classifier_only_arc_id=int(row.greedy_arc_id),
                        decoded_score=round(float(row.viterbi_prob), 6) if row.viterbi_prob == row.viterbi_prob else None,
                        classifier_only_score=round(float(row.greedy_prob), 6) if row.greedy_prob == row.greedy_prob else None,
                        decoder_status=str(row.decoder_status),
                        prediction_status=str(row.prediction_status),
                        gap_status=str(row.gap_status),
                        transition_hops=int(row.transition_hops),
                        is_connected_step=bool(row.is_connected_step),
                    )
                )

        total_wall_ms = round((time.perf_counter() - t_req_start) * 1000.0, 3)
        timing_out = dict(inf_out.timing_ms)
        timing_out["server_evaluation_ms"] = eval_res.evaluation_time_ms
        timing_out["total_request_ms"] = total_wall_ms

        logger.info(
            "POST /predict succeeded for trip_id='%s' in %.2f ms",
            trip_id,
            total_wall_ms,
        )
        logger.info("============================================================")

        per_obs_viterbi_arcs = [
            int(a) for a in inf_out.observation_predictions["viterbi_arc_id"].to_numpy(dtype=int)
        ]

        return PredictResponse(
            status="ok",
            trip_id=inf_out.trip_id,
            partition=inf_out.partition,
            model_name=inf_out.model_name,
            decoder="HMM + Viterbi",
            num_observations=int(inf_out.candidate_summary["num_observations"]),
            predicted_arc_ids=per_obs_viterbi_arcs,
            reconstructed_route_arcs=inf_out.predicted_route_arc_ids,
            predicted_route_arc_ids=inf_out.predicted_route_arc_ids,
            classifier_only_route_arc_ids=inf_out.classifier_only_route_arc_ids,
            predicted_route_segments_latlon=inf_out.predicted_route_segments_latlon,
            classifier_only_route_segments_latlon=inf_out.classifier_only_route_segments_latlon,
            gps_trajectory_latlon=inf_out.gps_trajectory_latlon,
            gps_points_latlon=inf_out.gps_trajectory_latlon,
            candidate_summary=inf_out.candidate_summary,
            decoder_summary=inf_out.decoder_summary,
            evaluation_metrics=eval_res.metrics,
            reference_evaluated_on_server=eval_res.reference_available,
            inference_time_ms=total_wall_ms,
            processing_time_ms=timing_out,
            observation_predictions=obs_records,
        )

    return app


app = create_app()
