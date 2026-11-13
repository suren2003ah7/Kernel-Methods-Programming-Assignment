import networkx as nx

def naive_multiplicate_graph_hopper(
        G: nx.Graph, 
        H: nx.Graph, 
        k_v: callable
) -> float:
    """
     naive Multiplicative GraphHopper kernel raphs.
    
    Parameters:
    - G: First networkx Graph
    - H: Second networkx Graph
    - k_v: A callable vertex kernel function taking two nodes (u, v) and returning a float similarity.
    
    Returns:
    - float: The computed kernel value.
    """

    def get_shortest_paths_by_length(graph):
        """Helper to collect all shortest paths grouped by their edge length l."""
        paths_by_len = {}
        nodes = list(graph.nodes())
        
        all_lengths = dict(nx.all_pairs_shortest_path_length(graph))
        
        for u in nodes:
            if u not in all_lengths:
                continue
            for v in nodes:
                if v in all_lengths[u]:
                    # Find all shortest paths between u and v
                    for path in nx.all_shortest_paths(graph, source=u, target=v):
                        l = len(path) - 1  # number of edges in the path
                        if l not in paths_by_len:
                            paths_by_len[l] = []
                        paths_by_len[l].append(path)
        return paths_by_len

    # Get shortest paths grouped by length for both graphs
    paths_G = get_shortest_paths_by_length(G)
    paths_H = get_shortest_paths_by_length(H)
    
    if not paths_G or not paths_H:
        return 0.0

    # find max diameter for both graphs
    max_l_G = max(paths_G.keys()) if paths_G else -1
    max_l_H = max(paths_H.keys()) if paths_H else -1
    delta = min(max_l_G, max_l_H)
    
    total_kernel_value = 0.0
    
    # Sum over all path lengths l from 0 to delta
    for l in range(delta + 1):
        if l not in paths_G or l not in paths_H:
            continue
            
        # Sum over all shortest paths of length l in G and H
        for pi in paths_G[l]:
            for pi_prime in paths_H[l]:
                # Compute the product of vertex similarities along the path
                path_product = 1.0
                for i in range(len(pi)):  # len(pi) is l + 1
                    u_node = pi[i]
                    v_node = pi_prime[i]
                    path_product *= k_v(u_node, v_node)
                
                total_kernel_value += path_product
                
    return float(total_kernel_value)


# small Test
if __name__ == "__main__":
    G1 = nx.Graph()
    G1.add_nodes_from([(0, {'label': 'A'}), (1, {'label': 'B'}), (2, {'label': 'C'})])
    G1.add_edges_from([(0, 1), (1, 2)])

    G2 = nx.Graph()
    G2.add_nodes_from([(0, {'label': 'A'}), (1, {'label': 'B'}), (2, {'label': 'C'})])
    G2.add_edges_from([(0, 1), (1, 2)])

    # Define a discrete vertex kernel
    def discrete_vertex_kernel(u, v):
        return 1.0 if G1.nodes[u].get('label') == G2.nodes[v].get('label') else 0.0

    result = naive_multiplicate_graph_hopper(G1, G2, discrete_vertex_kernel)
    print(f"Naive mgh Kernel: {result}")