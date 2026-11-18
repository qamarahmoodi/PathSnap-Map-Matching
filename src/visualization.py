"""Create static EDA plots and an interactive trip map."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium import FeatureGroup, LayerControl
from data_loader import load_trip

def create_plots(inventory_csv, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inventory_csv)
    if 'error' in df: df = df[df.error.isna()]
    for col, title, xlabel, filename in [
        ('gps_observations','GPS observations per trip','GPS observation count','gps_observations_per_trip.png'),
        ('max_timestamp_gap_s','Maximum timestamp gap per trip','Maximum gap (seconds)','max_timestamp_gap_per_trip.png'),
        ('road_nodes','Road nodes per trip','Road node count','road_nodes_per_trip.png'),
        ('road_arcs','Road arcs per trip','Road arc count','road_arcs_per_trip.png')]:
        plt.figure(figsize=(8,4.5))
        for part in ('train','student_test'):
            vals = df.loc[df.partition == part, col].dropna()
            if len(vals): plt.hist(vals, bins=20, alpha=.6, label=part)
        plt.title(title); plt.xlabel(xlabel); plt.ylabel('Trips'); plt.legend(); plt.tight_layout()
        plt.savefig(out/filename, dpi=160); plt.close()

def create_trip_map(trip_dir, output_html):
    d = load_trip(trip_dir); track,nodes,arcs,route = d['track'],d['nodes'],d['arcs'],d['route']
    lon,lat = track[:,0],track[:,1]
    m = folium.Map([float(lat.mean()),float(lon.mean())], zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='&copy; OpenStreetMap contributors &copy; CARTO',
        name='CartoDB Positron (Street Map)',
        subdomains='abcd',
        max_zoom=20,
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri',
        name='Esri World Street Map',
    ).add_to(m)
    folium.TileLayer(
        tiles='https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='&copy; OpenStreetMap contributors',
        name='OpenStreetMap',
        referrer_policy='origin',
    ).add_to(m)
    # Limit road graph rendering to a padded GPS bounding box for map responsiveness.
    pad_lon=max(float(np.ptp(lon))*.15,.001); pad_lat=max(float(np.ptp(lat))*.15,.001)
    mask=((nodes[arcs[:,0],0]+nodes[arcs[:,1],0])/2 >= lon.min()-pad_lon) & ((nodes[arcs[:,0],0]+nodes[arcs[:,1],0])/2 <= lon.max()+pad_lon) & ((nodes[arcs[:,0],1]+nodes[arcs[:,1],1])/2 >= lat.min()-pad_lat) & ((nodes[arcs[:,0],1]+nodes[arcs[:,1],1])/2 <= lat.max()+pad_lat)
    edges=np.where(mask)[0]
    if len(edges)>10000: edges=edges[::int(np.ceil(len(edges)/10000))]
    roads=FeatureGroup(name='Road network')
    for i in edges:
        a,b=arcs[i]; folium.PolyLine([[nodes[a,1],nodes[a,0]],[nodes[b,1],nodes[b,0]]],color='#9aa0a6',weight=1,opacity=.4).add_to(roads)
    roads.add_to(m)
    gps=FeatureGroup(name='Observed GPS')
    folium.PolyLine([[y,x] for x,y in zip(lon,lat)],color='blue',weight=4,tooltip='GPS observations').add_to(gps)
    gps.add_to(m)
    ref=FeatureGroup(name='Supplied reference route')
    valid=route[(route>=0)&(route<len(arcs))]
    for aid in valid:
        a,b=arcs[aid]; folium.PolyLine([[nodes[a,1],nodes[a,0]],[nodes[b,1],nodes[b,0]]],color='green',weight=4,opacity=.85).add_to(ref)
    ref.add_to(m); LayerControl().add_to(m)
    m.fit_bounds([[float(lat.min()-pad_lat),float(lon.min()-pad_lon)],[float(lat.max()+pad_lat),float(lon.max()+pad_lon)]])
    Path(output_html).parent.mkdir(parents=True,exist_ok=True); m.save(output_html)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inventory',default='outputs/dataset_inventory.csv'); ap.add_argument('--trip-dir',required=True); ap.add_argument('--out',default='outputs'); args=ap.parse_args()
    create_plots(args.inventory,args.out); create_trip_map(args.trip_dir,Path(args.out)/'trip_map.html')
    print('Plots and trip_map.html created in',args.out)
if __name__=='__main__': main()
