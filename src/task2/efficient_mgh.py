import networkx as nx

def efficient_multiplicate_graph_hopper(
        G: nx.Graph, 
        H: nx.Graph, 
        k_v: callable,
        edge_weight=None,
        vertex_label=None
) -> float:
    result = 0

    G_all_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=edge_weight))
    H_all_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(H, weight=edge_weight))

    G_shortest_path_dags = construct_shortest_path_dags(G, G_all_shortest_path_lengths, edge_weight)
    H_shortest_path_dags = construct_shortest_path_dags(H, H_all_shortest_path_lengths, edge_weight)

    for u in G.nodes():
        for v in H.nodes():
            if k_v(u, v) == 0:
                continue

            G_x_H = product_graph_dag(G_shortest_path_dags[u], H_shortest_path_dags[v], vertex_label)
            topological_order = list(nx.topological_sort(G_x_H))

            dp = {}
            dp[(u, v)] = k_v(u, v)

            for (x, y) in topological_order:
                if (x, y) == (u, v):
                    continue

                predecessor_contribution = sum(dp[(p_1, p_2)] for (p_1, p_2) in G_x_H.predecessors((x, y)))
                dp[(x, y)] = predecessor_contribution * k_v(x, y)

            result += sum(value for value in dp.values())

    return result

def construct_shortest_path_dags(
        G: nx.Graph, 
        shortest_path_lengths: dict,
        edge_weight=None
) -> dict:
    G_shortest_path_dags = {}

    for v in G.nodes():
        shortest_path_dag_v = nx.DiGraph()
        for u, w, edge_data in G.edges(data=True):
            weight_u_w = edge_data.get(edge_weight, 1)
            if (u in shortest_path_lengths[v] and w in shortest_path_lengths[v]):
                if (shortest_path_lengths[v][u] + weight_u_w == shortest_path_lengths[v][w]):
                    shortest_path_dag_v.add_edge(u, w)
                elif (not G.is_directed() and 
                    shortest_path_lengths[v][w] + weight_u_w == shortest_path_lengths[v][u]):
                    shortest_path_dag_v.add_edge(w, u)
        G_shortest_path_dags[v] = shortest_path_dag_v

    return G_shortest_path_dags

def product_graph_dag(G: nx.DiGraph, H: nx.DiGraph, vertex_label=None) -> nx.DiGraph:
    G_x_H = nx.DiGraph()

    for u, u_node_data in G.nodes(data=True):
        for v, v_node_data in H.nodes(data=True):
            if vertex_label is not None:
                if u_node_data.get(vertex_label) == v_node_data.get(vertex_label):
                    G_x_H.add_node((u, v))
            else:
                G_x_H.add_node((u, v))

    for (u,s) in G_x_H.nodes():
        neighbours = [(v, t) for v in G.neighbors(u) for t in H.neighbors(s) if (v, t) in G_x_H.nodes()]
        for (v, t) in neighbours:
            G_x_H.add_edge((u, s), (v, t))

    return G_x_H