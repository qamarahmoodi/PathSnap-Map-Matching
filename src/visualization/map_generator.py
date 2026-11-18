"""Reusable Folium Map Generation Pipeline for PathSnap.

Owner: L A Gokul (Common Evaluation & Reporting)

Creates interactive multi-layer Leaflet/Folium HTML maps visualizing:
1. Base Maps: CartoDB Positron, Esri World Street Map, OpenStreetMap.
2. Road Network Arcs (spatial bounding box filtered).
3. Raw GPS Trajectory (.track) with chronological point markers and timestamps.
4. Reference Ground-Truth Route (.route) when permitted.
5. Classifier-Only Greedy Predictions (highlighting Carriageway ping-pong & discontinuities).
6. Classifier + HMM/Viterbi Reconstructed Route (topologically continuous map-matched route).
7. Floating on-screen map legend and interactive LayerControl.

Correct coordinate ordering:
PathSnap data files store coordinates as [longitude, latitude].
Folium and Leaflet strictly require [latitude, longitude].
This module enforces rigorous conversion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import folium
from folium import FeatureGroup, LayerControl
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pathsnap.data.loader import load_trip, resolve_data_root
from pathsnap.graph.road_graph import build_trip_road_graph
from pathsnap.inference.inference_pipeline import run_trip_inference
from pathsnap.inference.model_scorer import CandidateModelScorer


def create_folium_map(
    trip_id: str,
    track: np.ndarray,            # Shape (N, 3): [lon, lat, timestamp]
    nodes: np.ndarray,            # Shape (num_nodes, 2): [lon, lat]
    arcs: np.ndarray,             # Shape (num_arcs, 2): [from_node, to_node]
    route: Optional[np.ndarray] = None,                       # Reference arc IDs
    classifier_preds: Optional[Sequence[int]] = None,         # Greedy arc IDs
    viterbi_preds: Optional[Sequence[int]] = None,            # HMM arc IDs
    output_html: Optional[Path] = None,
    padding_pct: float = 0.15,
) -> folium.Map:
    """Generate a publication-grade interactive Folium map for a single trip.
    
    Args:
        trip_id: Trip identifier string (e.g. "00000000").
        track: GPS trajectory array [lon, lat, timestamp].
        nodes: Road node coordinate array [lon, lat].
        arcs: Road arc endpoint array [from_node, to_node].
        route: Ground truth sequence of arc IDs (optional/permitted).
        classifier_preds: Classifier-only predicted arc sequence (optional).
        viterbi_preds: HMM/Viterbi reconstructed arc sequence (optional).
        output_html: Destination Path to save the interactive HTML map.
        padding_pct: Spatial bounding box padding around GPS points.
        
    Returns:
        folium.Map instance.
    """
    lon = track[:, 0]
    lat = track[:, 1]
    timestamps = track[:, 2] if track.shape[1] >= 3 else np.arange(len(track))

    center_lat = float(np.mean(lat))
    center_lon = float(np.mean(lon))

    # Initialize Folium Map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )

    # 1. Base Tile Layers
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Positron (Clean Light)",
        subdomains="abcd",
        max_zoom=20,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri World Street Map",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors",
        name="OpenStreetMap Standard",
        referrer_policy="origin",
    ).add_to(m)

    # Bounding Box Calculation for spatial clipping
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    d_lon = max(lon_max - lon_min, 0.002)
    d_lat = max(lat_max - lat_min, 0.002)
    bbox_lon_min, bbox_lon_max = lon_min - padding_pct * d_lon, lon_max + padding_pct * d_lon
    bbox_lat_min, bbox_lat_max = lat_min - padding_pct * d_lat, lat_max + padding_pct * d_lat

    # 2. Road Network Geometry Layer (filtered to bounding box)
    fg_roads = FeatureGroup(name="1. Road Network Arcs (.arcs)", show=True)
    num_arcs = len(arcs)
    
    # Fast midpoint vector filtering
    from_nodes = arcs[:, 0]
    to_nodes = arcs[:, 1]
    mid_lon = (nodes[from_nodes, 0] + nodes[to_nodes, 0]) / 2.0
    mid_lat = (nodes[from_nodes, 1] + nodes[to_nodes, 1]) / 2.0
    mask = (mid_lon >= bbox_lon_min) & (mid_lon <= bbox_lon_max) & (mid_lat >= bbox_lat_min) & (mid_lat <= bbox_lat_max)
    visible_arc_ids = np.where(mask)[0]

    # Subsample if arc density is overwhelmingly high (>12,000 arcs)
    if len(visible_arc_ids) > 12000:
        step = int(np.ceil(len(visible_arc_ids) / 12000))
        visible_arc_ids = visible_arc_ids[::step]

    for aid in visible_arc_ids:
        u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
        u_lat, u_lon = float(nodes[u, 1]), float(nodes[u, 0])
        v_lat, v_lon = float(nodes[v, 1]), float(nodes[v, 0])
        folium.PolyLine(
            locations=[[u_lat, u_lon], [v_lat, v_lon]],
            color="#94a3b8",
            weight=1.5,
            opacity=0.45,
            tooltip=f"Road Arc #{aid} ({u} → {v})",
        ).add_to(fg_roads)
    fg_roads.add_to(m)

    # 3. Raw GPS Trajectory Layer (.track)
    fg_gps = FeatureGroup(name="2. Raw GPS Trajectory (.track)", show=True)
    gps_coords = [[float(y), float(x)] for x, y in zip(lon, lat)]
    folium.PolyLine(
        locations=gps_coords,
        color="#2563eb",
        weight=3.5,
        opacity=0.8,
        tooltip=f"GPS Trajectory ({len(gps_coords)} points)",
    ).add_to(fg_gps)

    # Add start and end pins
    folium.CircleMarker(
        location=gps_coords[0],
        radius=7,
        color="#15803d",
        fill=True,
        fill_color="#22c55e",
        fill_opacity=0.9,
        popup=f"GPS Origin: t={timestamps[0]:.1f}s",
    ).add_to(fg_gps)

    folium.CircleMarker(
        location=gps_coords[-1],
        radius=7,
        color="#b91c1c",
        fill=True,
        fill_color="#ef4444",
        fill_opacity=0.9,
        popup=f"GPS Destination: t={timestamps[-1]:.1f}s",
    ).add_to(fg_gps)
    fg_gps.add_to(m)

    # 4. Reference Route Layer (.route)
    if route is not None and len(route) > 0:
        fg_ref = FeatureGroup(name="3. Reference Ground Truth (.route)", show=True)
        valid_route = [int(a) for a in route if 0 <= int(a) < num_arcs]
        for aid in valid_route:
            u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
            u_lat, u_lon = float(nodes[u, 1]), float(nodes[u, 0])
            v_lat, v_lon = float(nodes[v, 1]), float(nodes[v, 0])
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color="#16a34a",
                weight=6,
                opacity=0.75,
                tooltip=f"Reference Arc #{aid}",
            ).add_to(fg_ref)
        fg_ref.add_to(m)

    # 5. Classifier-Only Greedy Predictions Layer
    if classifier_preds is not None and len(classifier_preds) > 0:
        fg_greedy = FeatureGroup(name="4. Classifier-Only Greedy Top-1", show=False)
        for aid in classifier_preds:
            if 0 <= int(aid) < num_arcs:
                u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
                folium.PolyLine(
                    locations=[[float(nodes[u, 1]), float(nodes[u, 0])], [float(nodes[v, 1]), float(nodes[v, 0])]],
                    color="#ea580c",
                    weight=4,
                    opacity=0.85,
                    dash_array="6",
                    tooltip=f"Classifier Greedy Arc #{aid}",
                ).add_to(fg_greedy)
        fg_greedy.add_to(m)

    # 6. Classifier + HMM/Viterbi Reconstructed Route Layer
    if viterbi_preds is not None and len(viterbi_preds) > 0:
        fg_vit = FeatureGroup(name="5. HMM/Viterbi Reconstructed Route", show=True)
        for aid in viterbi_preds:
            if 0 <= int(aid) < num_arcs:
                u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
                folium.PolyLine(
                    locations=[[float(nodes[u, 1]), float(nodes[u, 0])], [float(nodes[v, 1]), float(nodes[v, 0])]],
                    color="#dc2626",
                    weight=4.5,
                    opacity=0.9,
                    tooltip=f"HMM/Viterbi Arc #{aid}",
                ).add_to(fg_vit)
        fg_vit.add_to(m)

    # 7. Add Floating On-Screen Map Legend
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 280px; height: auto;
        background-color: white; z-index:9999; font-size:12px;
        border:2px solid #334155; border-radius: 8px; padding: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: bold; color: #0f172a;">
            PathSnap Map: Trip {trip_id}
        </h4>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:16px; height:3px; background:#2563eb; vertical-align:middle; margin-right:6px;"></span>
            <b>Raw GPS (.track)</b>
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:16px; height:4px; background:#16a34a; vertical-align:middle; margin-right:6px;"></span>
            <b>Reference Route (.route)</b>
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:16px; height:3px; background:#ea580c; border-top:1px dashed #ea580c; vertical-align:middle; margin-right:6px;"></span>
            <b>Classifier-Only (Greedy)</b>
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:16px; height:4px; background:#dc2626; vertical-align:middle; margin-right:6px;"></span>
            <b>HMM/Viterbi Reconstructed</b>
        </div>
        <div>
            <span style="display:inline-block; width:16px; height:2px; background:#94a3b8; vertical-align:middle; margin-right:6px;"></span>
            Road Graph (.arcs)
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[bbox_lat_min, bbox_lon_min], [bbox_lat_max, bbox_lon_max]])

    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_html))

    return m


def generate_trip_map_from_disk(
    trip_id: str,
    partition: str = "student_test",
    output_html: Optional[Path] = None,
    model_scorer: Optional[CandidateModelScorer] = None,
) -> Path:
    """Load a trip from disk, run route-free inference, and output an interactive map."""
    raw = load_trip(str(trip_id), partition=partition, include_route=True)
    if model_scorer is None:
        model_scorer = CandidateModelScorer.from_path()

    inf_out = run_trip_inference(
        trip_id=str(trip_id),
        partition=partition,
        scorer=model_scorer,
    )

    if output_html is None:
        output_html = REPO_ROOT / "outputs" / "maps" / f"trip_{trip_id}_map.html"

    create_folium_map(
        trip_id=trip_id,
        track=raw.track[["longitude", "latitude", "timestamp_s"]].to_numpy(),
        nodes=raw.nodes,
        arcs=raw.arcs,
        route=raw.route,
        classifier_preds=inf_out.classifier_only_route_arc_ids,
        viterbi_preds=inf_out.predicted_route_arc_ids,
        output_html=output_html,
    )
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap Interactive Folium Map Generator")
    parser.add_argument("--trip-id", default="00000000", help="Trip ID (e.g. 00000000)")
    parser.add_argument("--partition", default="student_test", help="Partition (train or student_test)")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else REPO_ROOT / "outputs" / "maps" / f"trip_{args.trip_id}_map.html"
    print(f"Generating interactive Folium map for Trip {args.trip_id} ({args.partition})...")
    saved = generate_trip_map_from_disk(trip_id=args.trip_id, partition=args.partition, output_html=out_p)
    print(f"[SUCCESS] Interactive Folium map saved to: {saved}")


if __name__ == "__main__":
    main()
