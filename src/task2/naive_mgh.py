from collections import defaultdict
import networkx as nx

def naive_multiplicate_graph_hopper(
        G: nx.Graph, 
        H: nx.Graph, 
        k_v: callable,
        edge_weight: str = None
) -> float:
    """
    Parameters:
    - G: First networkx Graph
    - H: Second networkx Graph
    - k_v: A callable vertex kernel function taking two nodes (u, v) and returning a float similarity.
    - edge_weight: Name of the edge attribute to use as weight (None for unweighted graphs).
    
    Returns:
    - float: The computed kernel value.
    """

    def get_shortest_paths_by_length(graph):
        paths_by_len = defaultdict(list)
        all_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=edge_weight))
        
        for u, targets in all_lengths.items():
            for v in targets:
                for path in nx.all_shortest_paths(graph, source=u, target=v, weight=edge_weight):
                    l = len(path) - 1
                    paths_by_len[l].append(path)
        return paths_by_len

    paths_G = get_shortest_paths_by_length(G)
    paths_H = get_shortest_paths_by_length(H)
    
    if not paths_G or not paths_H:
        return 0.0

    delta = min(max(paths_G.keys()), max(paths_H.keys()))
    total_kernel_value = 0.0
    
    for l in range(delta + 1):
        for pi in paths_G[l]:
            for pi_prime in paths_H[l]:
                path_product = 1.0
                for i in range(len(pi)):
                    path_product *= k_v(pi[i], pi_prime[i])
                
                total_kernel_value += path_product
                
    return float(total_kernel_value)