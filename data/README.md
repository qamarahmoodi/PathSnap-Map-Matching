# PathSnap Dataset Directory & Inventory Guide

## Raw Dataset Location (Never Committed to Git)

The raw dataset resides outside this repository in the sibling directory:

```text
../Student folder/
├── train/           # 80 training trips
│   └── <TRIP_ID>/
│       ├── <TRIP_ID>.track
│       ├── <TRIP_ID>.nodes
│       ├── <TRIP_ID>.arcs
│       └── <TRIP_ID>.route
└── student_test/    # 10 development-stage test trips
    └── <TRIP_ID>/
        ├── <TRIP_ID>.track
        ├── <TRIP_ID>.nodes
        ├── <TRIP_ID>.arcs
        └── <TRIP_ID>.route
```

Configure the dataset path using the `PATHSNAP_DATA_ROOT` environment variable or pass `data_root` to `pathsnap.data.loader.resolve_data_root()`.

## Tracked Metadata Artifacts in `data/`

- `dataset_inventory.csv`: Full audit of all 80 training trips and 10 student-testing trips (observation counts, node counts, arc counts, route lengths, coordinate extents, timestamp intervals, and road connectivity validation).
- `sensitivity_analysis.csv`: Empirical sensitivity analysis of candidate recall and candidate miss rate across search radii $R \in \{30, 50, 75, 100, 150\}\text{ m}$ and candidate cap $K \in \{3, 5, 8, 10, 15, 20\}$ on training trips.
