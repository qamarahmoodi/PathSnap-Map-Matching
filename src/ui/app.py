"""Laptop 2 Evaluator User Interface for PathSnap (Two-Laptop Deployment Architecture).

Connects over the local network to Qamar's FastAPI server on Laptop 1 (`PATHSNAP_API_BASE_URL`).
Strictly enforces client-side evaluation isolation:
  - Never loads ML models or road-network files locally.
  - Never accesses or displays ground-truth `.route` files or reference labels.
  - Communicates exclusively via `GET /health`, `GET /trips`, and `POST /predict`.
  - Clearly separates Live API Mode from Mock Contract Preview Mode (`PATHSNAP_UI_MOCK_MODE`).
"""

from __future__ import annotations

import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import folium

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ui.api_client import (
    ENV_API_BASE_URL,
    ENV_UI_MOCK_MODE,
    PathSnapAPIClient,
    PredictResponse,
    get_configured_api_base_url,
    is_mock_mode_enabled,
)


DEFAULT_UI_HOST = "0.0.0.0"
DEFAULT_UI_PORT = int(os.environ.get("PATHSNAP_UI_PORT", "8501"))


def build_prediction_folium_map_html(pred: PredictResponse) -> str:
    """Render an interactive Folium HTML map showing the predicted route and raw GPS points (never reference route)."""
    all_lats = []
    all_lons = []

    for seg in pred.predicted_route_segments_latlon:
        for pt in seg:
            if len(pt) >= 2:
                all_lats.append(float(pt[0]))
                all_lons.append(float(pt[1]))

    for pt in pred.gps_points_latlon:
        if len(pt) >= 2:
            all_lats.append(float(pt[0]))
            all_lons.append(float(pt[1]))

    center_lat = sum(all_lats) / len(all_lats) if all_lats else 49.845
    center_lon = sum(all_lons) / len(all_lons) if all_lons else 8.595

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Positron",
        subdomains="abcd",
        max_zoom=20,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri World Street Map",
    ).add_to(m)

    if pred.gps_points_latlon:
        fg_gps = folium.FeatureGroup(name="Observed GPS Trajectory (.track)", show=True)
        folium.PolyLine(
            pred.gps_points_latlon,
            color="#2563eb",
            weight=3,
            opacity=0.65,
            dash_array="5",
            tooltip=f"Trip {pred.trip_id} Raw GPS Trajectory ({len(pred.gps_points_latlon)} pts)",
        ).add_to(fg_gps)
        fg_gps.add_to(m)

    if pred.predicted_route_segments_latlon:
        fg_pred = folium.FeatureGroup(name="Predicted Route (ML + HMM/Viterbi)", show=True)
        for idx, seg in enumerate(pred.predicted_route_segments_latlon):
            arc_lbl = (
                pred.reconstructed_route_arcs[idx]
                if idx < len(pred.reconstructed_route_arcs)
                else idx
            )
            folium.PolyLine(
                seg,
                color="#dc2626",
                weight=5,
                opacity=0.9,
                tooltip=f"Predicted Segment #{idx + 1} (Arc ID: {arc_lbl})",
            ).add_to(fg_pred)
        fg_pred.add_to(m)

    if all_lats and all_lons:
        m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()


def render_dashboard_page(
    api_base_url: str,
    mock_mode: bool,
    selected_trip_id: Optional[str] = None,
    run_prediction: bool = False,
) -> str:
    """Render the complete Laptop 2 Evaluator HTML Dashboard."""
    client = PathSnapAPIClient(base_url=api_base_url, mock_mode=mock_mode)
    health = client.check_health()
    trips_resp = client.get_trips()

    pred_resp: Optional[PredictResponse] = None
    if run_prediction and selected_trip_id:
        pred_resp = client.predict_trip(selected_trip_id)

    mode_badge = (
        '<span class="badge badge-mock">MOCK CONTRACT PREVIEW MODE (SIMULATED)</span>'
        if mock_mode
        else '<span class="badge badge-live">LIVE API MODE (LAPTOP 1 FASTAPI)</span>'
    )
    health_badge = (
        f'<span class="badge badge-ok">ONLINE ({health.latency_ms:.1f} ms) — {html.escape(health.status)}</span>'
        if health.ok
        else f'<span class="badge badge-err">OFFLINE — {html.escape(health.error_message or "Unreachable")}</span>'
    )

    # Build trip selector options
    trip_options_html = []
    if trips_resp.ok and trips_resp.trips:
        for t in trips_resp.trips:
            tid = str(t.get("trip_id", ""))
            lbl = str(t.get("label", f"Trip {tid}"))
            sel = "selected" if tid == selected_trip_id else ""
            trip_options_html.append(
                f'<option value="{html.escape(tid)}" {sel}>{html.escape(lbl)}</option>'
            )
    options_str = "\n".join(trip_options_html)

    # Build prediction result section
    result_section_html = ""
    if not trips_resp.ok:
        result_section_html = f"""
        <div class="alert alert-error">
            <strong>Error Retrieving Trips from Laptop 1 API (`GET /trips`):</strong><br/>
            <code>{html.escape(trips_resp.error_message or "Unknown error")}</code><br/>
            Verify that Qamar's FastAPI backend is running at <code>{html.escape(api_base_url)}</code> or switch to Mock Contract Preview Mode to inspect the UI layout.
        </div>
        """
    elif not trips_resp.trips:
        result_section_html = """
        <div class="alert alert-warn">
            <strong>Empty Trip List:</strong> <code>GET /trips</code> returned 0 evaluation trips.
        </div>
        """
    elif pred_resp is not None:
        if not pred_resp.ok:
            result_section_html = f"""
            <div class="alert alert-error">
                <strong>Prediction Error (`POST /predict` for Trip {html.escape(selected_trip_id or "")}):</strong><br/>
                <code>{html.escape(pred_resp.error_message or "Unknown error")}</code>
            </div>
            """
        else:
            mock_warning = (
                '<div class="alert alert-warn"><strong>MOCK MODE NOTICE:</strong> This prediction is generated from the deterministic mock contract adapter for UI verification. Switch to <strong>Live API Mode</strong> when connected to Laptop 1 for official evaluation.</div>'
                if pred_resp.is_mock
                else ""
            )
            metrics = pred_resp.evaluation_metrics or {}
            top1_str = f"{float(metrics['observation_top1_accuracy']) * 100:.2f}%" if "observation_top1_accuracy" in metrics else "Server-Side / Pending"
            f1_str = f"{float(metrics['macro_f1']):.4f}" if "macro_f1" in metrics else "Server-Side / Pending"
            cont_str = f"{float(metrics['route_continuity']) * 100:.2f}%" if "route_continuity" in metrics else "Server-Side / Pending"
            miss_str = f"{float(metrics['candidate_miss_rate']) * 100:.2f}%" if "candidate_miss_rate" in metrics else "Server-Side / Pending"

            map_iframe_html = ""
            if pred_resp.predicted_route_segments_latlon or pred_resp.gps_points_latlon:
                folium_src = build_prediction_folium_map_html(pred_resp)
                escaped_src = html.escape(folium_src, quote=True)
                map_iframe_html = f'<iframe srcdoc="{escaped_src}" style="width:100%;height:460px;border:1px solid #cbd5e1;border-radius:8px;"></iframe>'
            else:
                map_iframe_html = '<div class="alert alert-info">Predicted arc sequence returned without coordinate polyline geometry; displaying chronological reconstructed road arc IDs below.</div>'

            arc_preview = ", ".join(str(a) for a in pred_resp.reconstructed_route_arcs[:60])
            if len(pred_resp.reconstructed_route_arcs) > 60:
                arc_preview += f" ... (+{len(pred_resp.reconstructed_route_arcs) - 60} more segments)"

            result_section_html = f"""
            {mock_warning}
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                    <h2 style="margin:0;font-size:1.25rem;color:#0f172a;">
                        Reconstructed Route &amp; Server Evaluation — Trip <code>{html.escape(pred_resp.trip_id)}</code>
                    </h2>
                    <span class="badge badge-ok">POST /predict HTTP 200 ({pred_resp.inference_time_ms:.1f} ms)</span>
                </div>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-label">Observation Top-1 Accuracy</div>
                        <div class="metric-value">{html.escape(top1_str)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Candidate Macro F1</div>
                        <div class="metric-value">{html.escape(f1_str)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Viterbi Route Continuity</div>
                        <div class="metric-value">{html.escape(cont_str)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Candidate Miss Rate</div>
                        <div class="metric-value">{html.escape(miss_str)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">GPS Observations Matched</div>
                        <div class="metric-value">{pred_resp.num_observations:,}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Reconstructed Route Arcs</div>
                        <div class="metric-value">{len(pred_resp.reconstructed_route_arcs):,}</div>
                    </div>
                </div>
                <div style="margin-top:14px;">
                    {map_iframe_html}
                </div>
                <div style="margin-top:14px;background:#f8fafc;padding:12px;border-radius:6px;border:1px solid #e2e8f0;font-size:0.88rem;">
                    <strong>Model &amp; Sequential Decoder:</strong> {html.escape(pred_resp.model_name)} | {html.escape(pred_resp.decoder)}<br/>
                    <strong>Reconstructed Chronological Road Arc IDs ({len(pred_resp.reconstructed_route_arcs)} segments):</strong><br/>
                    <code style="word-break:break-all;color:#1e293b;">[{html.escape(arc_preview)}]</code>
                </div>
            </div>
            """
    else:
        result_section_html = """
        <div class="card" style="text-align:center;padding:36px;color:#475569;">
            <h3 style="margin-top:0;color:#1e293b;">Ready for Trip Selection</h3>
            <p>Select any of the 10 evaluation trips above and click <strong>Reconstruct Route (POST /predict)</strong> to send an inference request to Laptop 1.</p>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>PathSnap — Laptop 2 Evaluator Interface</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f1f5f9; color: #0f172a; margin: 0; padding: 0; }}
        .header {{ background: #0f172a; color: #ffffff; padding: 16px 28px; display: flex; justify-content: space-between; align-items: center; }}
        .container {{ max-width: 1180px; margin: 20px auto; padding: 0 20px; }}
        .card {{ background: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 18px; border: 1px solid #e2e8f0; }}
        .badge {{ padding: 5px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }}
        .badge-live {{ background: #2563eb; color: #fff; }}
        .badge-mock {{ background: #d97706; color: #fff; }}
        .badge-ok {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
        .badge-err {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
        .form-row {{ display: flex; gap: 14px; align-items: flex-end; flex-wrap: wrap; }}
        .form-group {{ display: flex; flex-direction: column; gap: 5px; flex: 1; min-width: 220px; }}
        label {{ font-size: 0.85rem; font-weight: 600; color: #334155; }}
        input[type="text"], select {{ padding: 9px 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.95rem; }}
        button {{ background: #2563eb; color: #ffffff; border: none; padding: 10px 20px; border-radius: 6px; font-size: 0.95rem; font-weight: 600; cursor: pointer; }}
        button:hover {{ background: #1d4ed8; }}
        .btn-secondary {{ background: #475569; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
        .metric-box {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 8px; }}
        .metric-label {{ font-size: 0.78rem; color: #64748b; text-transform: uppercase; font-weight: 600; }}
        .metric-value {{ font-size: 1.35rem; font-weight: 700; color: #0f172a; margin-top: 4px; }}
        .alert {{ padding: 14px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.92rem; }}
        .alert-error {{ background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }}
        .alert-warn {{ background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }}
        .alert-info {{ background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }}
    </style>
    <script>
        function showLoading() {{
            var btn = document.getElementById('predict-btn');
            if (btn) {{
                btn.innerText = 'Running Pipeline on Laptop 1...';
                btn.style.opacity = '0.75';
            }}
        }}
    </script>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0;font-size:1.35rem;">PathSnap — Distributed Map-Matching Evaluator UI (Laptop 2)</h1>
            <div style="font-size:0.82rem;color:#94a3b8;margin-top:3px;">
                Client-Side Zero-Leakage Interface | Connects to Laptop 1 FastAPI Server (`GET /health`, `GET /trips`, `POST /predict`)
            </div>
        </div>
        <div style="display:flex;gap:10px;align-items:center;">
            {mode_badge}
            {health_badge}
        </div>
    </div>

    <div class="container">
        <div class="card">
            <form method="GET" action="/" onsubmit="showLoading()">
                <div class="form-row">
                    <div class="form-group" style="flex:1.4;">
                        <label for="api_url">Laptop 1 FastAPI Base URL (<code>{ENV_API_BASE_URL}</code>)</label>
                        <input type="text" id="api_url" name="api_url" value="{html.escape(api_base_url)}" placeholder="http://192.168.1.10:8000"/>
                    </div>
                    <div class="form-group" style="flex:0.8;">
                        <label for="mode">Connection Mode</label>
                        <select id="mode" name="mode">
                            <option value="live" {"selected" if not mock_mode else ""}>Live API Mode (Laptop 1 Server)</option>
                            <option value="mock" {"selected" if mock_mode else ""}>Mock Contract Preview Mode</option>
                        </select>
                    </div>
                    <div class="form-group" style="flex:1.2;">
                        <label for="trip_id">Select Evaluation Trip ({len(trips_resp.trips)} options from <code>GET /trips</code>)</label>
                        <select id="trip_id" name="trip_id">
                            {options_str}
                        </select>
                    </div>
                    <div>
                        <button type="submit" id="predict-btn" name="action" value="predict">Reconstruct Route (POST /predict)</button>
                    </div>
                </div>
            </form>
        </div>

        {result_section_html}
    </div>
</body>
</html>"""


class EvaluatorUIRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the Laptop 2 Evaluator UI."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        api_url = qs.get("api_url", [get_configured_api_base_url()])[0]
        mode_param = qs.get("mode", [None])[0]
        if mode_param is not None:
            mock_mode = mode_param.lower() == "mock"
        else:
            mock_mode = is_mock_mode_enabled()

        selected_trip = qs.get("trip_id", [None])[0]
        run_pred = qs.get("action", [""])[0] == "predict" or (selected_trip is not None)

        page_html = render_dashboard_page(
            api_base_url=api_url,
            mock_mode=mock_mode,
            selected_trip_id=selected_trip,
            run_prediction=run_pred,
        )
        body = page_html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_ui_server(host: str = DEFAULT_UI_HOST, port: int = DEFAULT_UI_PORT) -> None:
    """Start the Laptop 2 Evaluator UI HTTP server."""
    server = HTTPServer((host, port), EvaluatorUIRequestHandler)
    print(f"PathSnap Laptop 2 Evaluator UI listening on http://{host}:{port}")
    print(f"Configured Laptop 1 API Base URL: {get_configured_api_base_url()}")
    print(f"Mock Mode Enabled: {is_mock_mode_enabled()}")
    server.serve_forever()


if __name__ == "__main__":
    run_ui_server()
