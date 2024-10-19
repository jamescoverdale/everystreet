from flask import Flask, request, jsonify, send_file, render_template
app = Flask(__name__)

import warnings
warnings.filterwarnings("ignore")

from libs.tools import *
from libs.graph_route import plot_graph_route

import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from shapely.geometry import Polygon
from network import Network
from network.algorithms import hierholzer
import shapely
import re

import io

@app.route('/')
def home():
   return render_template('index.html')

def coords_string_to_polygon(coords_string):
    # Extract all pairs of numbers using regex
    coords_pairs = re.findall(r'\[([-\d.]+),\s*([\d.]+)\]|\n?([-\d.]+),\s*([\d.]+)', coords_string)
    
    coords = []
    for pair in coords_pairs:
        if pair[0] and pair[1]:  # First pattern matched (bracketed format)
            lon, lat = float(pair[0]), float(pair[1])
        else:  # Second pattern matched (newline format)
            lon, lat = float(pair[2]), float(pair[3])
        coords.append((lon, lat))  # Note: Shapely expects (lat, lon) order
    
    # Create a Polygon (note: first and last point should match to close the polygon)
    if coords[0] != coords[-1]:
        coords.append(coords[0])  # Close the polygon by adding first point at the end
        
        
    return Polygon(coords)

@app.route('/api/polygon', methods=['POST'])
def create_item():

    content = request.get_data(as_text=True)

    ox.settings.use_cache = True
    ox.settings.log_console = True

    CUSTOM_FILTER = (
        '["highway"]["area"!~"yes"]["highway"!~"bridleway|bus_guideway|bus_stop|construction|cycleway|elevator|footway|'
        'motorway|motorway_junction|motorway_link|escalator|proposed|construction|platform|raceway|rest_area|'
        'path|service"]["access"!~"customers|no|private"]["public_transport"!~"platform"]'
        '["fee"!~"yes"]["foot"!~"no"]["service"!~"drive-through|driveway|parking_aisle"]["toll"!~"yes"]'
    )

    from shapely import Polygon
    import shapely
    p = coords_string_to_polygon(content)

    print(f"Polygon is valid: {p.is_valid}")
    print(f"Polygon area: {p.area}")
    print(f"Polygon coordinates: {list(p.exterior.coords)}")

    org_graph = ox.graph_from_polygon(p)

    # Simplifying the original directed multi-graph to undirected, so we can go both ways in one way streets
    graph = ox.convert.to_undirected(org_graph)
    fig, ax = ox.plot_graph(graph, node_zorder=2, node_color="k", bgcolor="w", show=False)

    # Finds the odd degree nodes and minimal matching
    odd_degree_nodes = get_odd_degree_nodes(graph)
    pair_weights = get_shortest_distance_for_odd_degrees(graph, odd_degree_nodes)
    matched_edges_with_weights = min_matching(pair_weights)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black', frameon=False)
    for v, u, w in matched_edges_with_weights:
        x = graph.nodes[v]["x"], graph.nodes[u]["x"]
        y = graph.nodes[v]["y"], graph.nodes[u]["y"]
        ax.plot(x, y, c='red', alpha=0.3)
        ax.scatter(x, y, c='red', edgecolor="none")

    fig, ax = ox.plot_graph(graph, node_zorder=2, node_color='g', bgcolor='k', ax=ax, show=False)

    # List all edges of the extended graph including original edges and edges from minimal matching
    single_edges = [(u, v) for u, v, k in graph.edges]
    added_edges = get_shortest_paths(graph, matched_edges_with_weights)
    edges = map_osmnx_edges2integers(graph, single_edges + added_edges)

    # Finds the Eulerian path
    network = Network(len(graph.nodes), edges, weighted=True)
    eulerian_path = hierholzer(network)
    converted_eulerian_path = convert_integer_path2osmnx_nodes(eulerian_path, graph.nodes())
    double_edge_heap = get_double_edge_heap(org_graph)

    # Finds the final path with edge IDs
    final_path = convert_path(graph, converted_eulerian_path, double_edge_heap)

    fig, ax = plot_graph_route(org_graph, final_path, route_linewidth=6, node_size=0, bgcolor="w", route_alpha=0.2, route_color="w")

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)