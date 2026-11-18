"""PathSnap dataset inventory and integrity checks."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_trip, iter_trip_dirs


def inspect_trip(partition, trip_dir):
    d = load_trip(trip_dir)
    track, nodes, arcs, route = d['track'], d['nodes'], d['arcs'], d['route']
    lon, lat, ts = track.T
    dt = np.diff(ts)
    arc_endpoints_ok = ((arcs >= 0) & (arcs < len(nodes))).all(axis=1)
    route_ids_ok = (route >= 0) & (route < len(arcs))
    valid_route = route[route_ids_ok]
    if len(valid_route) > 1:
        continuity = float((arcs[valid_route[:-1], 1] == arcs[valid_route[1:], 0]).mean())
    else:
        continuity = float('nan')
    positive_dt = dt[dt > 0]
    return {
        'partition': partition, 'trip_id': trip_dir.name,
        'gps_observations': len(track), 'road_nodes': len(nodes),
        'road_arcs': len(arcs), 'reference_route_arcs': len(route),
        'nonfinite_track_rows': int((~np.isfinite(track).all(axis=1)).sum()),
        'invalid_lonlat_rows': int(((np.abs(lon) > 180) | (np.abs(lat) > 90)).sum()),
        'non_increasing_timestamp_steps': int((dt <= 0).sum()),
        'median_timestamp_step_s': float(np.median(positive_dt)) if len(positive_dt) else np.nan,
        'max_timestamp_gap_s': float(np.max(dt)) if len(dt) else np.nan,
        'gaps_over_1_5s': int((dt > 1.5).sum()),
        'invalid_arc_endpoint_rows': int((~arc_endpoints_ok).sum()),
        'invalid_reference_arc_ids': int((~route_ids_ok).sum()),
        'reference_route_connectivity': continuity,
        'longitude_min': float(np.min(lon)), 'longitude_max': float(np.max(lon)),
        'latitude_min': float(np.min(lat)), 'latitude_max': float(np.max(lat)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='Folder containing train/ and student_test/')
    ap.add_argument('--out', default='outputs')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in ('train', 'student_test'):
        for trip in iter_trip_dirs(args.data_root, part):
            try: rows.append(inspect_trip(part, trip))
            except Exception as e: rows.append({'partition': part, 'trip_id': trip.name, 'error': str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(out/'dataset_inventory.csv', index=False)
    good = df[df.get('error', pd.Series(index=df.index, dtype=object)).isna()]
    lines = ['PathSnap dataset EDA summary', f'Data root: {Path(args.data_root).resolve()}', '']
    for part in ('train', 'student_test'):
        g = good[good.partition == part]
        lines += [f'{part}: {len(g)} trips']
        if len(g):
            for col in ('gps_observations','road_nodes','road_arcs','reference_route_arcs'):
                lines.append(f'  total {col}: {int(g[col].sum())}')
            lines.append(f"  trips with gaps > 1.5s: {int((g.gaps_over_1_5s > 0).sum())}")
            lines.append(f"  trips with invalid GPS rows: {int(((g.nonfinite_track_rows + g.invalid_lonlat_rows) > 0).sum())}")
            lines.append(f"  mean reference-route connectivity: {g.reference_route_connectivity.mean():.5f}")
            lines.append(f"  maximum timestamp gap: {g.max_timestamp_gap_s.max():.3f}s")
        lines.append('')
    (out/'partition_summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print(f'Inventory written to {out / "dataset_inventory.csv"}')

if __name__ == '__main__': main()
