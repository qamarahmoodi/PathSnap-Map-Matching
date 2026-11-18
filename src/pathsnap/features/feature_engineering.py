"""GPS kinematics and candidate-level spatial feature engineering for PathSnap."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pathsnap.schemas.candidate_schema import CandidateConfig
from pathsnap.spatial.projection import angular_difference_deg, compute_bearing_deg


@dataclass
class GPSKinematics:
    """Observation-level kinematic features derived solely from causal GPS trajectory steps."""

    pts_xy_m: np.ndarray        # float64, shape (N_obs, 2)
    delta_time_s: np.ndarray    # float32, shape (N_obs,), NaN for obs 0 or invalid dt
    displacement_m: np.ndarray  # float32, shape (N_obs,), NaN for obs 0
    gps_speed_mps: np.ndarray   # float32, shape (N_obs,), NaN when dt invalid or > max_valid_delta_time_s
    gps_bearing_deg: np.ndarray # float32, shape (N_obs,), NaN when displacement < min_displacement_for_bearing_m


def compute_gps_kinematics(
    pts_xy_m: np.ndarray,
    timestamps_s: np.ndarray,
    config: CandidateConfig,
) -> GPSKinematics:
    """Calculate causal backward-difference time steps, speeds, and headings without future leakage.

    Rules:
    - Observation 0 has NaN for `delta_time_s`, `gps_speed_mps`, and `gps_bearing_deg`.
    - If `dt <= 0` (duplicate or non-monotonic timestamp), `delta_time_s`, `gps_speed_mps`, and
      `gps_bearing_deg` are set to NaN.
    - If `dt > config.max_valid_delta_time_s`, `delta_time_s` retains the positive elapsed time gap
      (so downstream models and sequence decoders can detect GPS dropouts), while `gps_speed_mps` and
      `gps_bearing_deg` are set to NaN so long dropouts never fabricate artificial velocities.
    - If step displacement `< config.min_displacement_for_bearing_m` (e.g., stationary jitter),
      `gps_bearing_deg` is set to NaN because direction of static GPS noise is meaningless.
    """
    n_obs = pts_xy_m.shape[0]
    delta_time = np.full(n_obs, np.nan, dtype=np.float32)
    displacement = np.full(n_obs, np.nan, dtype=np.float32)
    speed = np.full(n_obs, np.nan, dtype=np.float32)
    bearing = np.full(n_obs, np.nan, dtype=np.float32)

    if n_obs <= 1:
        return GPSKinematics(
            pts_xy_m=pts_xy_m,
            delta_time_s=delta_time,
            displacement_m=displacement,
            gps_speed_mps=speed,
            gps_bearing_deg=bearing,
        )

    dt_raw = np.diff(timestamps_s.astype(np.float64))
    dx = np.diff(pts_xy_m[:, 0])
    dy = np.diff(pts_xy_m[:, 1])
    disp_raw = np.hypot(dx, dy)

    delta_time[1:] = np.where(dt_raw > 0.0, dt_raw, np.nan).astype(np.float32)
    displacement[1:] = disp_raw.astype(np.float32)

    valid_dt = (dt_raw > 0.0) & (dt_raw <= config.max_valid_delta_time_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        spd_raw = np.where(valid_dt, disp_raw / dt_raw, np.nan)

    # Filter out physically implausible telemetry spikes
    spd_raw = np.where(spd_raw <= config.max_plausible_speed_mps, spd_raw, np.nan)
    speed[1:] = spd_raw.astype(np.float32)

    valid_bearing = (
        valid_dt
        & np.isfinite(spd_raw)
        & (disp_raw >= config.min_displacement_for_bearing_m)
    )
    brg_raw = np.where(valid_bearing, compute_bearing_deg(dx, dy), np.nan)
    bearing[1:] = brg_raw.astype(np.float32)

    return GPSKinematics(
        pts_xy_m=pts_xy_m,
        delta_time_s=delta_time,
        displacement_m=displacement,
        gps_speed_mps=speed,
        gps_bearing_deg=bearing,
    )


def compute_candidate_bearing_difference(
    gps_bearing_deg: np.ndarray,
    segment_bearing_deg: np.ndarray,
) -> np.ndarray:
    """Compute angular difference in [0, 180] degrees, preserving NaN where gps_bearing_deg is NaN."""
    out = np.full(segment_bearing_deg.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(gps_bearing_deg)
    if np.any(valid):
        out[valid] = angular_difference_deg(
            gps_bearing_deg[valid].astype(np.float64),
            segment_bearing_deg[valid].astype(np.float64),
        ).astype(np.float32)
    return out
