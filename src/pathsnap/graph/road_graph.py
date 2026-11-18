"""HMM-compatible directed road graph representation for a single trip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class TripRoadGraph:
    """Directed road network graph for a single trip, used by HMM/Viterbi decoding.

    Attributes:
        trip_id: Preserve exact trip identifier string (with leading zeros).
        arc_endpoints: int32 array of shape (num_arcs, 2) with [from_node_id, to_node_id].
        arc_lengths_m: float32 array of shape (num_arcs,) in metres.
        arc_bearings_deg: float32 array of shape (num_arcs,) in [0, 360) degrees clockwise from North.
        outgoing_arcs_by_node: Mapping from node_id (int) to 1D int32 numpy array of outgoing arc_ids.
    """

    trip_id: str
    arc_endpoints: np.ndarray
    arc_lengths_m: np.ndarray
    arc_bearings_deg: np.ndarray
    outgoing_arcs_by_node: Dict[int, np.ndarray]

    def __post_init__(self) -> None:
        self.arc_endpoints = np.asarray(self.arc_endpoints, dtype=np.int32)
        self.arc_lengths_m = np.asarray(self.arc_lengths_m, dtype=np.float32)
        self.arc_bearings_deg = np.asarray(self.arc_bearings_deg, dtype=np.float32)
        if self.arc_endpoints.ndim != 2 or self.arc_endpoints.shape[1] != 2:
            raise ValueError("arc_endpoints must have shape (num_arcs, 2)")
        num_arcs = self.arc_endpoints.shape[0]
        if self.arc_lengths_m.shape != (num_arcs,) or self.arc_bearings_deg.shape != (num_arcs,):
            raise ValueError("arc_lengths_m and arc_bearings_deg must have shape (num_arcs,)")

    @property
    def num_arcs(self) -> int:
        return int(self.arc_endpoints.shape[0])

    def successor_arcs(self, arc_id: int) -> np.ndarray:
        """Return 1D int32 array of arc IDs whose starting node equals the ending node of `arc_id`."""
        if arc_id < 0 or arc_id >= self.num_arcs:
            raise IndexError(f"arc_id {arc_id} out of bounds for graph with {self.num_arcs} arcs")
        end_node = int(self.arc_endpoints[arc_id, 1])
        res = self.outgoing_arcs_by_node.get(end_node)
        if res is None:
            return np.empty(0, dtype=np.int32)
        return res

    def are_connected(self, from_arc_id: int, to_arc_id: int) -> bool:
        """Return True if `from_arc_id` is identical to `to_arc_id` or directly connects into `to_arc_id`."""
        if from_arc_id == to_arc_id:
            return True
        return int(self.arc_endpoints[from_arc_id, 1]) == int(self.arc_endpoints[to_arc_id, 0])

    def shortest_hop_transition_type(self, from_arc_id: int, to_arc_id: int, max_hops: int = 3) -> int:
        """Return number of arc transitions between `from_arc_id` and `to_arc_id` up to `max_hops` (-1 if unreachable).

        0: same arc (stay on segment)
        1: direct successor (end_node(from) == start_node(to))
        k <= max_hops: k intermediate hops
        -1: not reachable within max_hops
        """
        if from_arc_id == to_arc_id:
            return 0
        target_start_node = int(self.arc_endpoints[to_arc_id, 0])
        curr_end_node = int(self.arc_endpoints[from_arc_id, 1])
        if curr_end_node == target_start_node:
            return 1

        visited_nodes = {curr_end_node}
        frontier = [curr_end_node]
        for hop in range(2, max_hops + 1):
            next_frontier: List[int] = []
            for node in frontier:
                out_arcs = self.outgoing_arcs_by_node.get(node)
                if out_arcs is None:
                    continue
                for next_arc in out_arcs:
                    next_end = int(self.arc_endpoints[int(next_arc), 1])
                    if next_end == target_start_node:
                        return hop
                    if next_end not in visited_nodes:
                        visited_nodes.add(next_end)
                        next_frontier.append(next_end)
            if not next_frontier:
                break
            frontier = next_frontier
        return -1


def build_trip_road_graph(
    trip_id: str,
    arcs: np.ndarray,
    arc_lengths_m: np.ndarray,
    arc_bearings_deg: np.ndarray,
) -> TripRoadGraph:
    """Build a TripRoadGraph with O(1) outgoing-arc lookup per node."""
    arcs_arr = np.asarray(arcs, dtype=np.int32)
    num_arcs = arcs_arr.shape[0]
    from_nodes = arcs_arr[:, 0]

    # Sort arc indices by from_node to group efficiently without Python loops over every arc
    order = np.argsort(from_nodes, kind="stable")
    sorted_from = from_nodes[order]
    unique_nodes, split_idx = np.unique(sorted_from, return_index=True)
    groups = np.split(order.astype(np.int32), split_idx[1:])

    outgoing_arcs_by_node: Dict[int, np.ndarray] = {
        int(node): grp for node, grp in zip(unique_nodes, groups)
    }

    return TripRoadGraph(
        trip_id=str(trip_id),
        arc_endpoints=arcs_arr,
        arc_lengths_m=np.asarray(arc_lengths_m, dtype=np.float32),
        arc_bearings_deg=np.asarray(arc_bearings_deg, dtype=np.float32),
        outgoing_arcs_by_node=outgoing_arcs_by_node,
    )
