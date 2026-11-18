"""Detailed data-quality analysis for PathSnap trips.

Run from the project root:
python src/quality_analysis.py --data-root "C:\\path\\to\\Student folder" --out outputs

Expected data-root structure:
    Student folder/
        train/<trip_id>/
        student_test/<trip_id>/

Uses src/data_loader.py's load_trip(trip_dir) function.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow execution as `python src/quality_analysis.py` from the project root.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_trip


def as_2d(array, columns, name):
    """Normalize empty/single-row arrays to a 2-D shape."""
    arr = np.asarray(array)
    if arr.size == 0:
        return np.empty((0, columns), dtype=float)
    if arr.ndim == 1:
        if arr.size == columns:
            return arr.reshape(1, columns)
        raise ValueError(f"{name} expected {columns} columns, got shape {arr.shape}")
    if arr.ndim != 2 or arr.shape[1] != columns:
        raise ValueError(f"{name} expected N x {columns}, got shape {arr.shape}")
    return arr


def count_nonfinite(values):
    try:
        arr = np.asarray(values, dtype=float)
        return int((~np.isfinite(arr)).sum())
    except (TypeError, ValueError):
        return int(pd.isna(np.asarray(values)).sum())


def weak_graph_stats(node_count, arcs):
    """Return weakly connected component count and isolated node count."""
    if node_count == 0:
        return 0, 0
    parent = list(range(node_count))
    rank = [0] * node_count

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    degree = np.zeros(node_count, dtype=np.int64)
    valid = 0
    for row in arcs:
        try:
            a, b = int(row[0]), int(row[1])
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= a < node_count and 0 <= b < node_count:
            union(a, b)
            degree[a] += 1
            degree[b] += 1
            valid += 1

    components = len({find(i) for i in range(node_count)})
    isolated = int((degree == 0).sum())
    return components, isolated


def analyze_trip(trip_dir, partition, trip_id):
    loaded = load_trip(trip_dir)
    track = as_2d(loaded["track"], 3, "track")
    nodes = as_2d(loaded["nodes"], 2, "nodes")
    arcs = as_2d(loaded["arcs"], 2, "arcs")
    route = np.asarray(loaded["route"]).reshape(-1)

    # Per-column non-finite/missing counts.
    columns = {
        "track_longitude": track[:, 0],
        "track_latitude": track[:, 1],
        "track_timestamp": track[:, 2],
        "node_longitude": nodes[:, 0],
        "node_latitude": nodes[:, 1],
        "arc_from_node": arcs[:, 0],
        "arc_to_node": arcs[:, 1],
        "route_arc_id": route,
    }
    missing = {
        "partition": partition,
        "trip_id": trip_id,
        "track_rows": len(track),
        "node_rows": len(nodes),
        "arc_rows": len(arcs),
        "route_rows": len(route),
    }
    for name, values in columns.items():
        missing[f"{name}_nonfinite"] = count_nonfinite(values)

    # GPS coordinate checks.
    gps_finite = np.isfinite(track[:, :2]).all(axis=1) if len(track) else np.array([], dtype=bool)
    gps_valid_range = (
        gps_finite
        & (track[:, 0] >= -180) & (track[:, 0] <= 180)
        & (track[:, 1] >= -90) & (track[:, 1] <= 90)
    ) if len(track) else gps_finite
    node_finite = np.isfinite(nodes).all(axis=1) if len(nodes) else np.array([], dtype=bool)
    node_valid_range = (
        node_finite
        & (nodes[:, 0] >= -180) & (nodes[:, 0] <= 180)
        & (nodes[:, 1] >= -90) & (nodes[:, 1] <= 90)
    ) if len(nodes) else node_finite

    # Timestamp statistics: differences are calculated in the stored timestamp unit.
    timestamps = track[:, 2].astype(float) if len(track) else np.array([], dtype=float)
    finite_ts = timestamps[np.isfinite(timestamps)]
    gaps = np.diff(finite_ts) if len(finite_ts) > 1 else np.array([], dtype=float)
    positive_gaps = gaps[gaps > 0]
    timestamp_non_increasing = int((gaps <= 0).sum())
    timestamp_gap_max = float(np.max(positive_gaps)) if len(positive_gaps) else np.nan
    timestamp_gap_mean = float(np.mean(positive_gaps)) if len(positive_gaps) else np.nan
    timestamp_gap_median = float(np.median(positive_gaps)) if len(positive_gaps) else np.nan
    gaps_gt_1_5 = int((positive_gaps > 1.5).sum())

    # Validate arc endpoints against per-trip node row indices.
    arc_finite = np.isfinite(arcs).all(axis=1) if len(arcs) else np.array([], dtype=bool)
    arc_integer = np.zeros(len(arcs), dtype=bool)
    if len(arcs):
        arc_integer = arc_finite & np.isclose(arcs, np.round(arcs)).all(axis=1)
    arc_idx_valid = np.zeros(len(arcs), dtype=bool)
    if len(arcs):
        safe = arc_integer
        if safe.any():
            endpoints = np.round(arcs[safe]).astype(np.int64)
            arc_idx_valid[safe] = (
                (endpoints[:, 0] >= 0) & (endpoints[:, 0] < len(nodes))
                & (endpoints[:, 1] >= 0) & (endpoints[:, 1] < len(nodes))
            )
    invalid_arc_endpoints = int((~arc_idx_valid).sum())

    # Validate route IDs against this trip's arc row indices.
    route_numeric = pd.to_numeric(pd.Series(route), errors="coerce").to_numpy(dtype=float)
    route_finite = np.isfinite(route_numeric)
    route_integer = route_finite & np.isclose(route_numeric, np.round(route_numeric))
    route_valid = route_integer & (route_numeric >= 0) & (route_numeric < len(arcs))
    invalid_route_ids = int((~route_valid).sum())

    # Ordered route continuity: end node of arc i should equal start node of arc i+1.
    valid_route_ids = route_numeric[route_valid].astype(np.int64)
    route_arc_rows = arcs[valid_route_ids] if len(valid_route_ids) else np.empty((0, 2))
    route_pairs_valid = np.isfinite(route_arc_rows).all(axis=1) if len(route_arc_rows) else np.array([], dtype=bool)
    route_pairs_valid &= np.isclose(route_arc_rows, np.round(route_arc_rows)).all(axis=1) if len(route_arc_rows) else route_pairs_valid
    if len(route_arc_rows):
        safe_rows = route_arc_rows[route_pairs_valid].astype(np.int64)
        safe_pair_mask = (
            (safe_rows[:, 0] >= 0) & (safe_rows[:, 0] < len(nodes))
            & (safe_rows[:, 1] >= 0) & (safe_rows[:, 1] < len(nodes))
        )
        safe_rows = safe_rows[safe_pair_mask]
    else:
        safe_rows = np.empty((0, 2), dtype=np.int64)
    route_discontinuities = int((safe_rows[:-1, 1] != safe_rows[1:, 0]).sum()) if len(safe_rows) > 1 else 0
    route_transition_count = max(len(safe_rows) - 1, 0)

    components, isolated_nodes = weak_graph_stats(len(nodes), arcs[arc_idx_valid] if len(arcs) else arcs)

    quality = {
        "partition": partition,
        "trip_id": trip_id,
        "gps_invalid_coordinate_rows": int((~gps_valid_range).sum()),
        "node_invalid_coordinate_rows": int((~node_valid_range).sum()),
        "invalid_arc_endpoint_rows": invalid_arc_endpoints,
        "invalid_route_arc_ids": invalid_route_ids,
        "timestamp_min": float(np.min(finite_ts)) if len(finite_ts) else np.nan,
        "timestamp_max": float(np.max(finite_ts)) if len(finite_ts) else np.nan,
        "timestamp_positive_gap_count": int(len(positive_gaps)),
        "timestamp_gap_mean_stored_units": timestamp_gap_mean,
        "timestamp_gap_median_stored_units": timestamp_gap_median,
        "timestamp_gap_max_stored_units": timestamp_gap_max,
        "timestamp_gaps_gt_1_5_stored_units": gaps_gt_1_5,
        "timestamp_non_increasing_transitions": timestamp_non_increasing,
        "weakly_connected_components": components,
        "isolated_nodes": isolated_nodes,
        "reference_route_discontinuities": route_discontinuities,
        "reference_route_transitions_checked": route_transition_count,
    }
    return missing, quality


def main():
    parser = argparse.ArgumentParser(description="Run detailed PathSnap data-quality analysis.")
    parser.add_argument("--data-root", required=True, help="Folder containing train and student_test.")
    parser.add_argument("--out", default="outputs", help="Output folder (default: outputs).")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_rows, quality_rows, errors = [], [], []
    for partition in ("train", "student_test"):
        partition_dir = data_root / partition
        if not partition_dir.is_dir():
            raise FileNotFoundError(f"Expected partition folder not found: {partition_dir}")
        trip_dirs = sorted(p for p in partition_dir.iterdir() if p.is_dir())
        for trip_dir in trip_dirs:
            try:
                missing, quality = analyze_trip(trip_dir, partition, trip_dir.name)
                missing_rows.append(missing)
                quality_rows.append(quality)
            except Exception as exc:
                errors.append({
                    "partition": partition,
                    "trip_id": trip_dir.name,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    missing_df = pd.DataFrame(missing_rows)
    quality_df = pd.DataFrame(quality_rows)
    errors_df = pd.DataFrame(errors, columns=["partition", "trip_id", "error"])

    missing_df.to_csv(output_dir / "missing_value_report.csv", index=False)
    quality_df.to_csv(output_dir / "data_quality_report.csv", index=False)
    errors_df.to_csv(output_dir / "quality_analysis_errors.csv", index=False)

    summary_lines = [
        "PathSnap detailed data-quality analysis",
        f"Trips analyzed successfully: {len(missing_df)}",
        f"Trips with loader/analysis errors: {len(errors_df)}",
        "",
        "Totals across successfully analyzed trips:",
    ]
    if len(missing_df):
        count_cols = [c for c in missing_df.columns if c.endswith("_nonfinite")]
        for col in count_cols:
            summary_lines.append(f"{col}: {int(missing_df[col].sum())}")
    if len(quality_df):
        for col in (
            "gps_invalid_coordinate_rows", "node_invalid_coordinate_rows",
            "invalid_arc_endpoint_rows", "invalid_route_arc_ids",
            "timestamp_gaps_gt_1_5_stored_units",
            "timestamp_non_increasing_transitions", "isolated_nodes",
            "reference_route_discontinuities",
        ):
            summary_lines.append(f"{col}: {int(quality_df[col].sum())}")
        summary_lines += [
            "",
            "Notes:",
            "- Timestamp gap units are the same as the timestamp units stored in .track.",
            "- The >1.5 threshold is applied directly to stored timestamp differences.",
            "- Graph components are weakly connected components (arc direction ignored).",
            "- Route continuity checks whether consecutive valid route arcs share an end/start node.",
            "- student_test is profiled for data quality only; it is not used for model training.",
        ]
    (output_dir / "data_quality_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Analyzed {len(missing_df)} trips; {len(errors_df)} errors.")
    print(f"Reports written to: {output_dir.resolve()}")
    print(" - missing_value_report.csv")
    print(" - data_quality_report.csv")
    print(" - quality_analysis_errors.csv")
    print(" - data_quality_summary.txt")


if __name__ == "__main__":
    main()
