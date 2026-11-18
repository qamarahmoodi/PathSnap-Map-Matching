# Dataset exploration — PathSnap

## Dataset structure
Describe the `.track`, `.nodes`, `.arcs`, and `.route` files using the supplied brief. Note that graph identifiers are local to each trip.

## Dataset inventory
Insert the generated `outputs/dataset_inventory.csv` summary here. Report the number of trips and totals separately for `train` and `student_test`.

## Data-quality checks
Discuss non-finite GPS rows, coordinate ranges, timestamp ordering, large timestamp gaps, invalid arc endpoints, invalid route IDs, and route connectivity. Use measured results from `outputs/partition_summary.txt`; do not estimate or fabricate values.

## Visual exploration
Insert the GPS observation, timestamp-gap, node-count, and arc-count histograms. Link the interactive Folium map and explain its three layers: local road graph, observed GPS trajectory, and supplied reference route.

## Limitations and handoff
The reference route is not row-aligned to GPS observations. Document the label-generation approach agreed with the map-matching and ML leads. Do not claim the supplied route is error-free ground truth.
