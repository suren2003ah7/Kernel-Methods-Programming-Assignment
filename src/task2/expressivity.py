from itertools import product

import networkx as nx
import grakel as gk
import matplotlib.pyplot as plt
from efficient_mgh import efficient_multiplicate_graph_hopper


def create_binary_labeled_graphs(graphs, vertex_labels=("A", "B")):
    labeled_graphs = []
    for graph in graphs:
        nodes = list(graph.nodes)
        for label_assignment in product(vertex_labels, repeat=len(nodes)):
            labeled_graph = graph.copy()
            nx.set_node_attributes(
                labeled_graph,
                dict(zip(nodes, label_assignment)),
                "label",
            )
            labeled_graphs.append(labeled_graph)
    return labeled_graphs


def expressivity_analysis_multiplicative_graph_hopper():
    graphs = nx.graph_atlas_g()[1:209]   # all graphs with up to 6 nodes

    g = list(filter(lambda g: nx.is_connected(g), graphs))
    labeled_graphs = create_binary_labeled_graphs(g)

    print(len(labeled_graphs), "labeled graphs created")

    for i in range(20):
        print(labeled_graphs[i].nodes(data=True))


if __name__ == "__main__":
    expressivity_analysis_multiplicative_graph_hopper()