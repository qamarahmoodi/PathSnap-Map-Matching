"""Tests for TripRoadGraph and successor arc lookups."""

import numpy as np

from pathsnap.data.loader import load_trip
from pathsnap.spatial.road_network import build_trip_road_network


def test_trip_road_graph_successors_and_route_connectivity():
    raw = load_trip("00000001", partition="train", include_route=True)
    net = build_trip_road_network(raw.trip_id, raw.nodes, raw.arcs)
    graph = net.graph

    assert graph.trip_id == "00000001"
    assert graph.arc_endpoints.dtype == np.int32
    assert graph.arc_lengths_m.dtype == np.float32
    assert graph.arc_bearings_deg.dtype == np.float32
    assert graph.num_arcs == raw.arcs.shape[0]

    # Verify successor_arcs definition: start_node(succ) == end_node(arc)
    for sample_arc in [0, 10, 100, int(raw.route[0]), int(raw.route[10])]:
        succs = graph.successor_arcs(sample_arc)
        end_node = int(raw.arcs[sample_arc, 1])
        for s in succs:
            assert int(raw.arcs[int(s), 0]) == end_node

    # Verify consecutive arcs in the reference route are connected in TripRoadGraph
    route = raw.route
    connected_steps = 0
    for k in range(len(route) - 1):
        a_from = int(route[k])
        a_to = int(route[k + 1])
        if graph.are_connected(a_from, a_to):
            connected_steps += 1
            if a_from != a_to:
                assert a_to in graph.successor_arcs(a_from)
    assert connected_steps == len(route) - 1
