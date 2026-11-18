"""Dataset location, trip file loading, and data-integrity validation for PathSnap."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


ENV_DATA_ROOT = "PATHSNAP_DATA_ROOT"


@dataclass
class TripRawData:
    """Raw arrays loaded for a single trip."""

    trip_id: str
    partition: str
    track: pd.DataFrame          # columns: observation_id, longitude, latitude, timestamp_s
    nodes: np.ndarray            # float64, shape (num_nodes, 2) -> [longitude, latitude]
    arcs: np.ndarray             # int32, shape (num_arcs, 2) -> [from_node_id, to_node_id]
    route: Optional[np.ndarray]  # int32, shape (num_route_arcs,) or None if not available/loaded


def resolve_data_root(data_root: Optional[str | Path] = None) -> Path:
    """Locate the dataset root directory containing `train/` and `student_test/`.

    Resolution order:
    1. Explicit `data_root` parameter if provided.
    2. `PATHSNAP_DATA_ROOT` environment variable.
    3. Relative sibling path `../Student folder` from current working directory or repo root.
    """
    candidates: List[Path] = []
    if data_root is not None:
        candidates.append(Path(data_root))
    env_val = os.environ.get(ENV_DATA_ROOT)
    if env_val:
        candidates.append(Path(env_val))

    cwd = Path.cwd()
    candidates.extend([
        cwd / ".." / "Student folder",
        cwd / "Student folder",
        Path(__file__).resolve().parents[4] / "Student folder",
    ])

    for cand in candidates:
        resolved = cand.resolve()
        if resolved.is_dir():
            # Check if nested "Student folder/Student folder" exists
            if (resolved / "train").is_dir() and (resolved / "student_test").is_dir():
                return resolved
            nested = resolved / "Student folder"
            if nested.is_dir() and (nested / "train").is_dir():
                return nested.resolve()

    raise FileNotFoundError(
        f"Could not locate PathSnap dataset root containing 'train/' and 'student_test/'. "
        f"Set {ENV_DATA_ROOT} or pass data_root explicitly."
    )


def list_trips(
    partition: str = "train",
    data_root: Optional[str | Path] = None,
) -> List[str]:
    """Return sorted list of trip_id strings (preserving leading zeros) in a partition."""
    root = resolve_data_root(data_root)
    part_dir = root / partition
    if not part_dir.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {part_dir}")
    trip_ids = sorted(
        [d.name for d in part_dir.iterdir() if d.is_dir()],
        key=lambda x: str(x),
    )
    return trip_ids


def load_track_file(path: Path) -> pd.DataFrame:
    """Load a `.track` file (whitespace-separated: longitude, latitude, timestamp_s)."""
    if not path.is_file():
        raise FileNotFoundError(f"Track file not found: {path}")
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in {path}, got shape {arr.shape}")
    df = pd.DataFrame(
        {
            "observation_id": np.arange(arr.shape[0], dtype=np.int32),
            "longitude": arr[:, 0],
            "latitude": arr[:, 1],
            "timestamp_s": arr[:, 2],
        }
    )
    return df


def load_nodes_file(path: Path) -> np.ndarray:
    """Load a `.nodes` file into a float64 array of shape (num_nodes, 2) [longitude, latitude]."""
    if not path.is_file():
        raise FileNotFoundError(f"Nodes file not found: {path}")
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {path}, got shape {arr.shape}")
    return arr


def load_arcs_file(path: Path) -> np.ndarray:
    """Load a `.arcs` file into an int32 array of shape (num_arcs, 2) [from_node_id, to_node_id]."""
    if not path.is_file():
        raise FileNotFoundError(f"Arcs file not found: {path}")
    arr = np.loadtxt(path, dtype=np.int32, ndmin=2)
    if arr.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {path}, got shape {arr.shape}")
    return arr


def load_route_file(path: Path) -> np.ndarray:
    """Load a `.route` file into a 1D int32 array of traversed arc IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Route file not found: {path}")
    arr = np.loadtxt(path, dtype=np.int32, ndmin=1)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1 column in {path}, got shape {arr.shape}")
    return arr


def validate_trip_arrays(
    trip_id: str,
    track: pd.DataFrame,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Validate coordinate ranges, node/arc references, timestamp ordering, and route connectivity."""
    num_obs = len(track)
    num_nodes = nodes.shape[0]
    num_arcs = arcs.shape[0]

    # Coordinate checks
    lon = track["longitude"].to_numpy()
    lat = track["latitude"].to_numpy()
    ts = track["timestamp_s"].to_numpy()

    non_finite_gps = int((~np.isfinite(lon) | ~np.isfinite(lat) | ~np.isfinite(ts)).sum())
    invalid_lon_lat = int(((np.abs(lon) > 180.0) | (np.abs(lat) > 90.0)).sum())

    # Timestamp checks
    dt = np.diff(ts) if num_obs > 1 else np.empty(0, dtype=np.float64)
    non_monotonic_ts = int((dt < 0).sum())
    zero_dt_count = int((dt == 0).sum())
    exact_1s_count = int(np.isclose(dt, 1.0, atol=1e-4).sum())
    gap_gt_1s_count = int((dt > 1.0 + 1e-4).sum())
    gap_gt_5s_count = int((dt > 5.0).sum())
    gap_gt_30s_count = int((dt > 30.0).sum())
    max_dt_s = float(dt.max()) if dt.size > 0 else 0.0
    min_dt_s = float(dt.min()) if dt.size > 0 else 0.0

    # Duplicate consecutive GPS coords
    if num_obs > 1:
        dup_coords = int(((np.diff(lon) == 0.0) & (np.diff(lat) == 0.0)).sum())
    else:
        dup_coords = 0

    # Arc reference checks
    invalid_arc_nodes = int(((arcs < 0) | (arcs >= num_nodes)).any(axis=1).sum())
    self_loop_arcs = int((arcs[:, 0] == arcs[:, 1]).sum())

    # Isolated nodes check
    referenced_nodes = np.unique(arcs.ravel())
    isolated_nodes_count = int(num_nodes - referenced_nodes.size)

    # Route checks (only if route provided)
    route_len = 0
    invalid_route_arcs = 0
    disconnected_route_transitions = 0
    repeated_route_arcs = 0
    if route is not None:
        route_len = int(route.size)
        invalid_Mask = (route < 0) | (route >= num_arcs)
        invalid_route_arcs = int(invalid_Mask.sum())
        if invalid_route_arcs == 0 and route_len > 1:
            end_nodes = arcs[route[:-1], 1]
            start_nodes = arcs[route[1:], 0]
            # Disconnected if consecutive route arcs are neither consecutive in node graph nor the same arc
            disconnected_route_transitions = int(((end_nodes != start_nodes) & (route[:-1] != route[1:])).sum())
        repeated_route_arcs = int(route_len - np.unique(route).size)

    return {
        "trip_id": str(trip_id),
        "num_observations": num_obs,
        "num_nodes": num_nodes,
        "num_arcs": num_arcs,
        "num_route_arcs": route_len,
        "min_lon": float(lon.min()),
        "max_lon": float(lon.max()),
        "min_lat": float(lat.min()),
        "max_lat": float(lat.max()),
        "duration_s": float(ts[-1] - ts[0]) if num_obs > 1 else 0.0,
        "non_finite_gps": non_finite_gps,
        "invalid_lon_lat": invalid_lon_lat,
        "non_monotonic_ts": non_monotonic_ts,
        "zero_dt_count": zero_dt_count,
        "exact_1s_ratio": float(exact_1s_count / dt.size) if dt.size > 0 else 1.0,
        "gap_gt_1s_count": gap_gt_1s_count,
        "gap_gt_5s_count": gap_gt_5s_count,
        "gap_gt_30s_count": gap_gt_30s_count,
        "min_dt_s": min_dt_s,
        "max_dt_s": max_dt_s,
        "dup_consecutive_coords": dup_coords,
        "invalid_arc_nodes": invalid_arc_nodes,
        "self_loop_arcs": self_loop_arcs,
        "isolated_nodes_count": isolated_nodes_count,
        "invalid_route_arcs": invalid_route_arcs,
        "disconnected_route_transitions": disconnected_route_transitions,
        "repeated_route_arcs": repeated_route_arcs,
    }


def load_trip(
    trip_id: str,
    partition: str = "train",
    data_root: Optional[str | Path] = None,
    include_route: bool = True,
) -> TripRawData:
    """Load `.track`, `.nodes`, `.arcs`, and optionally `.route` for a trip.

    Set `include_route=False` during inference to enforce strict training/inference separation.
    """
    trip_str = str(trip_id)
    root = resolve_data_root(data_root)
    trip_dir = root / partition / trip_str
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    track = load_track_file(trip_dir / f"{trip_str}.track")
    nodes = load_nodes_file(trip_dir / f"{trip_str}.nodes")
    arcs = load_arcs_file(trip_dir / f"{trip_str}.arcs")

    route: Optional[np.ndarray] = None
    if include_route:
        route_path = trip_dir / f"{trip_str}.route"
        if route_path.is_file():
            route = load_route_file(route_path)

    # Validate node IDs in arcs
    if ((arcs < 0) | (arcs >= nodes.shape[0])).any():
        raise ValueError(f"Trip {trip_str} contains out-of-bounds node IDs in .arcs")

    return TripRawData(
        trip_id=trip_str,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
    )
