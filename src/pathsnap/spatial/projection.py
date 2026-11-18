"""Geographic coordinate projection and metric calculations for PathSnap.

Converts angular WGS84 (longitude, latitude) coordinates in degrees into a local
metric Cartesian plane (Eastings x_m, Northings y_m) in metres.

Why local metric projection?
- Raw Euclidean distance in degrees distorts East-West vs North-South distances by cos(latitude).
  At latitude ~49.85 deg N (Germany), cos(49.85 deg) ≈ 0.6448, so 1 deg of longitude (~71.8 km) is
  35.5% shorter than 1 deg of latitude (~111.2 km).
- Using a local tangent-plane metric projection (equirectangular/orthographic around trip centroid
  or WGS84 ellipsoidal radii of curvature) achieves sub-centimetre accuracy over trip extents (< 30 km)
  and allows exact vectorised planar dot/cross products for segment projections and bearings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


# WGS84 ellipsoid constants
WGS84_A = 6378137.0               # Semi-major axis (m)
WGS84_F = 1.0 / 298.257223563     # Flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # First eccentricity squared


@dataclass(frozen=True)
class LocalTangentProjector:
    """Projects (longitude, latitude) in degrees to local (x_east_m, y_north_m) in metres
    using WGS84 meridional (M) and prime-vertical (N) radii of curvature at reference latitude.
    """

    lon0_deg: float
    lat0_deg: float
    meters_per_deg_lon: float
    meters_per_deg_lat: float

    @classmethod
    def from_lon_lat(cls, lon_deg: np.ndarray, lat_deg: np.ndarray) -> "LocalTangentProjector":
        """Create a local metric projector centred on the mean (lon, lat) of the trip."""
        lon0 = float(np.nanmean(lon_deg))
        lat0 = float(np.nanmean(lat_deg))
        lat0_rad = np.deg2rad(lat0)
        sin_lat = np.sin(lat0_rad)
        cos_lat = np.cos(lat0_rad)

        denom = np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        # Prime vertical radius of curvature N(phi)
        n_rad = WGS84_A / denom
        # Meridional radius of curvature M(phi)
        m_rad = WGS84_A * (1.0 - WGS84_E2) / (denom ** 3)

        m_per_deg_lon = float(np.deg2rad(1.0) * n_rad * cos_lat)
        m_per_deg_lat = float(np.deg2rad(1.0) * m_rad)

        return cls(
            lon0_deg=lon0,
            lat0_deg=lat0,
            meters_per_deg_lon=m_per_deg_lon,
            meters_per_deg_lat=m_per_deg_lat,
        )

    def project(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert (lon_deg, lat_deg) to (x_east_m, y_north_m) in float64 metres."""
        lon_arr = np.asarray(lon_deg, dtype=np.float64)
        lat_arr = np.asarray(lat_deg, dtype=np.float64)
        x_m = (lon_arr - self.lon0_deg) * self.meters_per_deg_lon
        y_m = (lat_arr - self.lat0_deg) * self.meters_per_deg_lat
        return x_m, y_m

    def project_xy(self, lon_lat: np.ndarray) -> np.ndarray:
        """Convert array of shape (..., 2) [lon, lat] into (..., 2) [x_east_m, y_north_m]."""
        arr = np.asarray(lon_lat, dtype=np.float64)
        out = np.empty_like(arr, dtype=np.float64)
        out[..., 0] = (arr[..., 0] - self.lon0_deg) * self.meters_per_deg_lon
        out[..., 1] = (arr[..., 1] - self.lat0_deg) * self.meters_per_deg_lat
        return out


def compute_bearing_deg(dx_east_m: np.ndarray, dy_north_m: np.ndarray) -> np.ndarray:
    """Compute azimuth bearing in [0, 360) degrees clockwise from True North (0=N, 90=E, 180=S, 270=W)."""
    angle_rad = np.arctan2(dx_east_m, dy_north_m)
    bearing_deg = np.rad2deg(angle_rad) % 360.0
    return bearing_deg


def angular_difference_deg(b1_deg: np.ndarray, b2_deg: np.ndarray) -> np.ndarray:
    """Compute smallest unsigned angular difference in [0, 180] degrees between two bearings."""
    diff = np.abs(b1_deg - b2_deg) % 360.0
    return np.minimum(diff, 360.0 - diff)
