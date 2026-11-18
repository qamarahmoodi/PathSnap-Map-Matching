"""Load PathSnap trip files. Arc and node IDs are local to each trip."""
from pathlib import Path
import numpy as np

EXTS = ('track', 'nodes', 'arcs', 'route')

def load_trip(trip_dir):
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    out = {}
    for ext in EXTS:
        path = trip_dir / f'{trip_id}.{ext}'
        if not path.exists():
            raise FileNotFoundError(path)
        dtype = int if ext in ('arcs', 'route') else float
        out[ext] = np.loadtxt(path, dtype=dtype, ndmin=2)
    if out['track'].shape[1] != 3:
        raise ValueError(f'{trip_id}: track must have 3 columns (lon, lat, timestamp)')
    if out['nodes'].shape[1] != 2:
        raise ValueError(f'{trip_id}: nodes must have 2 columns (lon, lat)')
    if out['arcs'].shape[1] != 2:
        raise ValueError(f'{trip_id}: arcs must have 2 columns (from_node, to_node)')
    out['route'] = out['route'].reshape(-1).astype(int)
    out['arcs'] = out['arcs'].astype(int)
    return out

def iter_trip_dirs(data_root, partition):
    folder = Path(data_root) / partition
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing partition folder: {folder}')
    yield from sorted(p for p in folder.iterdir() if p.is_dir())
