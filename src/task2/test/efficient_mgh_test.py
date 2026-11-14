import networkx as nx
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from efficient_mgh import efficient_multiplicate_graph_hopper
from naive_mgh import naive_multiplicate_graph_hopper

# --- Graph G: Diamond structure with a central split ---
#       (1: C)
#      /      \
# (0: C)      (3: O)
#      \      /
#       (2: C)
#         |
#       (4: N)
G = nx.Graph()
G.add_nodes_from([
    (0, {"label": "C"}),
    (1, {"label": "C"}),
    (2, {"label": "C"}),
    (3, {"label": "O"}),
    (4, {"label": "N"})
])
G.add_edges_from([
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (3, 4)
])

# --- Graph H: Similar diamond structure, with slightly different weights/labels ---
#       (10: C)
#      /       \
# (12: C)      (13: O)
#      \       /
#       (11: C)
#         |
#       (14: N)
H = nx.Graph()
H.add_nodes_from([
    (12, {"label": "C"}),
    (10, {"label": "C"}),
    (11, {"label": "C"}),
    (13, {"label": "O"}),
    (14, {"label": "N"})
])
H.add_edges_from([
    (12, 10),
    (12, 11),
    (10, 13),
    (11, 13),
    (13, 14)
])

def semantic_similarity(u, v):
    u_label = G.nodes[u].get("label")
    v_label = H.nodes[v].get("label")

    return 1.0 if u_label == v_label else 0.0

naive_score = naive_multiplicate_graph_hopper(
    G=G,
    H=H,
    k_v=semantic_similarity
)

efficient_score = efficient_multiplicate_graph_hopper(
    G=G,
    H=H,
    k_v=semantic_similarity,
    vertex_label="label"
)

print(f"Test Naive Graphhopper Kernel Score: {naive_score}")
print(f"Test Efficient Graphhopper Kernel Score: {efficient_score}")