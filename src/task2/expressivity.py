"""Find labeled graphs separated only by multiplicative GraphHopper."""
import argparse
from collections import Counter, defaultdict
from itertools import product
from math import cos, sin, tau
from pathlib import Path

import networkx as nx

from naive_mgh import naive_multiplicate_graph_hopper


def create_binary_labeled_graphs(graphs, vertex_labels=("A", "B")):
    for graph in graphs:
        nodes = list(graph)
        for assignment in product(vertex_labels, repeat=len(nodes)):
            labeled = graph.copy()
            nx.set_node_attributes(labeled, dict(zip(nodes, assignment)), "label")
            yield labeled


def features(graph):
    """Count ordered endpoint pairs, individual path positions, and full paths."""
    labels = nx.get_node_attributes(graph, "label")
    shortest_path, graphhopper, multiplicative = Counter(), Counter(), Counter()
    for start, distances in nx.all_pairs_shortest_path_length(graph):
        for end, distance in distances.items():
            if start != end:
                shortest_path[distance, labels[start], labels[end]] += 1
            for path in nx.all_shortest_paths(graph, start, end):
                sequence = tuple(labels[node] for node in path)
                multiplicative[sequence] += 1
                for position, label in enumerate(sequence):
                    graphhopper[distance, position, label] += 1
    return shortest_path, graphhopper, multiplicative


def signature(counter):
    return tuple(sorted(counter.items()))


def squared_distance(left, right):
    return sum((left[key] - right[key]) ** 2 for key in left.keys() | right.keys())


def distances(left, right):
    return tuple(squared_distance(a, b) for a, b in zip(features(left), features(right)))


def naive_score(left, right):
    return naive_multiplicate_graph_hopper(
        left, right, lambda u, v: int(left.nodes[u]["label"] == right.nodes[v]["label"])
    )


def check_against_naive(left, right, expected):
    observed = naive_score(left, left) + naive_score(right, right) - 2 * naive_score(left, right)
    if observed != expected:
        raise RuntimeError(f"Naive kernel distance {observed} differs from feature distance {expected}")


def labeled_graph(edges, labels):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(labels)))
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, dict(enumerate(labels)), "label")
    return graph


def verify_theory():
    edges = [(0, 1), (2, 3)]
    mixed, same = labeled_graph(edges, "ABAB"), labeled_graph(edges, "AABB")
    square = nx.disjoint_union_all([nx.cycle_graph(4)] + [nx.path_graph(2)] * 4)
    lines = nx.disjoint_union_all([nx.path_graph(3)] * 4)
    for graph in (square, lines):
        nx.set_node_attributes(graph, "A", "label")
    edges = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4), (2, 3)]
    known = labeled_graph(edges, "BAABBA"), labeled_graph(edges, "ABABBA")
    for name, left, right, expected in (
        ("2(a) two edges", mixed, same, (16, 0, 16)),
        ("2(a) square and lines", square, lines, (16, 0, 0)),
        ("2(d) reference pair", *known, (0, 0, 28)),
    ):
        observed = distances(left, right)
        if observed != expected:
            raise RuntimeError(f"{name}: expected {expected}, observed {observed}")
        check_against_naive(left, right, expected[2])
        print(f"{name}: {observed}")


def save_drawing(left, right, output):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 360">',
             '<rect width="680" height="360" fill="white"/>']
    for graph, name, center in ((left, "G", 170), (right, "H", 510)):
        layout = nx.circular_layout(graph) if False else None
        # Position vertices in sorted order around a circle without extra dependencies.
        from math import cos, sin, tau
        nodes = sorted(graph)
        pos = {u: (center + 110*cos(tau*i/len(nodes)), 170 + 110*sin(tau*i/len(nodes)))
               for i, u in enumerate(nodes)}
        parts.append(f'<text x="{center}" y="30" text-anchor="middle" font-size="20">{name}</text>')
        for u, v in graph.edges:
            x, y = pos[u]
            a, b = pos[v]
            parts.append(f'<line x1="{x}" y1="{y}" x2="{a}" y2="{b}" stroke="#334155" stroke-width="3"/>')
        for u, (x, y) in pos.items():
            label = graph.nodes[u]["label"]
            fill = "#bfdbfe" if label == "A" else "#fed7aa"
            parts.append(f'<circle cx="{x}" cy="{y}" r="19" fill="{fill}" stroke="#334155"/>')
            parts.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="13">{u}:{label}</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def search(output, example_limit):
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(dict)
    graphs = [g for g in nx.graph_atlas_g() if 1 <= len(g) <= 6 and nx.is_connected(g)]
    assignment_count = 0
    for graph in create_binary_labeled_graphs(graphs):
        assignment_count += 1
        sp, gh, mgh = features(graph)
        groups[signature(sp), signature(gh)].setdefault(signature(mgh), (graph, mgh))

    matching_groups = [variants for variants in groups.values() if len(variants) > 1]
    lines = [
        "# Task 2(d) — Expressivity search",
        "",
        f"We tested {assignment_count} binary label assignments across {len(graphs)} connected atlas graphs with one to six vertices. Assignments equivalent under relabeling were allowed during generation. We grouped graphs by their shortest-path and ordinary GraphHopper features, then found {len(matching_groups)} groups with more than one multiplicative feature vector.",
        "",
        "Shortest-path features count each ordered pair of distinct reachable endpoints once. Ordinary GraphHopper counts labels at individual positions of shortest paths. Multiplicative GraphHopper counts entire label sequences. Both GraphHopper counters include length zero. Distances below are squared differences between the count vectors.",
    ]
    for index, variants in enumerate(matching_groups[:example_limit], 1):
        (left, left_counts), (right, right_counts) = list(variants.values())[:2]
        result = distances(left, right)
        check_against_naive(left, right, result[2])
        sequence = next(key for key in sorted(left_counts.keys() | right_counts.keys())
                        if left_counts[key] != right_counts[key])
        name = f"pair_{index}.svg"
        save_drawing(left, right, output / name)
        lines.extend([
            "", f"## Pair {index}", "", f"![Labeled graph pair {index}]({name})", "",
            f"Squared distances in the order shortest-path, ordinary GraphHopper, and multiplicative GraphHopper are **{result}**.",
            f"The sequence **{'–'.join(sequence)}** occurs {left_counts[sequence]} time(s) in G and {right_counts[sequence]} time(s) in H.",
            f"Edges in G: {sorted(left.edges())}. Edges in H: {sorted(right.edges())}.",
        ])
    lines.extend([
        "", "## Checks against Task 2(a)", "",
        "The two-edge example gave distances (16, 0, 16), confirming that ordinary GraphHopper can miss a distinction captured by the multiplicative version. The square-and-lines example gave (16, 0, 0), confirming that the shortest-path kernel can distinguish graphs with equal multiplicative features. These disconnected examples were checked separately from the atlas search.",
        "",
        "For discrete labels, the general implication from Task 2(a) follows because the ordinary GraphHopper counts can be recovered by summing the full sequence counts at each position. The atlas search illustrates this comparison on a finite collection.",
        "",
    ])
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{assignment_count} assignments, {len(matching_groups)} qualifying baseline groups")
    print(f"Results saved to {output / 'results.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("task2d_results"))
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be at least 1")
    verify_theory()
    search(args.output, args.examples)
    expressivity_analysis_multiplicative_graph_hopper()
