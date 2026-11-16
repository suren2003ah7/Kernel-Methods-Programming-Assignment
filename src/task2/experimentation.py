import csv
import time
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from grakel.datasets import fetch_dataset
from grakel.kernels import GraphHopper, ShortestPath, VertexHistogram, WeisfeilerLehman
from grakel.utils import graph_from_networkx
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from efficient_mgh import efficient_multiplicate_graph_hopper
from naive_mgh import naive_multiplicate_graph_hopper

SEED = 42
C_VALUES = [10.0 ** p for p in range(-3, 4)]   # {1e-3, ..., 1e3}
WL_H_VALUES = [1, 2, 3, 4, 5]                  # h=0 is VertexHistogram
GAMMA_VALUES = [1e-3, 1e-2]
MAX_GRAPHS_MAP = {
    "MUTAG": 60,
    "PTC_FM": 60,
    "ENZYMES": 60,       # 6 classes x 10-fold CV requires >= 60 graphs
    "PROTEINS_full": 60,
}
SCALING_SIZES = [10, 20, 30, 40, 50]



# Data prep helpers
def select_indices(data, labels, max_n, max_nodes=None, min_per_class=1, seed=SEED):
    """Return stratified graph indices (optionally preferring graphs with |V| <= max_nodes)."""
    labels = np.asarray(labels)
    if max_n is None and max_nodes is None:
        return list(range(len(data)))

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    chosen = []

    for c, cnt in zip(classes, counts):
        idx = np.where(labels == c)[0]
        k = len(idx) if max_n is None else max(min_per_class, int(round(max_n * cnt / len(labels))))
        k = min(k, len(idx))

        if max_nodes is not None:
            small = [i for i in idx if 2 <= len(data[i][1]) <= max_nodes]
            picked = rng.choice(small, size=k, replace=False) if len(small) >= k else sorted(idx, key=lambda i: len(data[i][1]))[:k]
        else:
            picked = rng.choice(idx, size=k, replace=False)
        chosen.extend(picked)

    if max_n is not None and min_per_class <= 1:
        chosen = chosen[:max_n]
    return sorted(chosen)


def to_nx(item, disc_item=None):
    # grakel -> nx
    G = nx.Graph(item[0])
    G.add_nodes_from(item[1].keys())
    G.remove_edges_from(nx.selfloop_edges(G))
    if disc_item is None:
        nx.set_node_attributes(G, item[1], "label")
    else:
        nx.set_node_attributes(G, item[1], "attr")
        nx.set_node_attributes(G, disc_item[1], "label")
    G = nx.convert_node_labels_to_integers(G)
    if disc_item is not None:
        G.graph["attr_mat"] = np.asarray([G.nodes[u]["attr"] for u in range(G.number_of_nodes())], dtype=float)
    return G


def to_grakel_onehot(data):
    labels = sorted({lab for g in data for lab in g[1].values()})
    idx = {lab: i for i, lab in enumerate(labels)}
    eye = np.eye(len(labels), dtype=float).tolist()
    return [[g[0], {u: eye[idx[lab]] for u, lab in g[1].items()}] for g in data]



# Helpers:
def discrete_kappa_builder(G, H):
    lg, lh = dict(G.nodes(data="label")), dict(H.nodes(data="label"))
    return lambda u, v: 1.0 if lg[u] == lh[v] else 0.0

# for continuous
def gaussian_kappa_builder(gamma):
    def build(G, H):
        X, Y = G.graph["attr_mat"], H.graph["attr_mat"]
        sq = np.sum(X ** 2, axis=1, keepdims=True) + np.sum(Y ** 2, axis=1, keepdims=True).T - 2.0 * (X @ Y.T)
        M = np.exp(-gamma * np.maximum(sq, 0.0))
        lg = np.array([G.nodes[u]["label"] for u in range(G.number_of_nodes())])
        lh = np.array([H.nodes[v]["label"] for v in range(H.number_of_nodes())])
        M *= (lg[:, None] == lh[None, :])
        return lambda u, v: float(M[u, v])
    return build


def normalize_gram(K):
    diag = np.sqrt(np.maximum(np.diag(K), 0.0))
    denom = np.outer(diag, diag)
    return np.divide(K, denom, out=np.zeros_like(K, dtype=float), where=denom > 0)


def compute_kmgh_gram(nx_graphs, kappa_builder):
    n = len(nx_graphs)
    K = np.zeros((n, n), dtype=float)
    t0 = time.perf_counter()
    with tqdm(total=n * (n + 1) // 2, desc="Kmgh", unit="pair", leave=False) as pbar:
        for i in range(n):
            for j in range(i, n):
                val = float(efficient_multiplicate_graph_hopper(nx_graphs[i], nx_graphs[j], kappa_builder(nx_graphs[i], nx_graphs[j])))
                K[i, j] = K[j, i] = val
                pbar.update(1)
    return K, time.perf_counter() - t0


def compute_grakel_gram(kernel_obj, grakel_data):
    t0 = time.perf_counter()
    K = np.nan_to_num(np.asarray(kernel_obj.fit_transform(grakel_data), dtype=float), nan=0.0)
    return 0.5 * (K + K.T), time.perf_counter() - t0 # the 0.5 * (K + K.T)'s here to enforce symmetry in case of rounding errors


class PrecomputedKernelSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, gram_matrices, C=1.0, k_idx=0):
        self.gram_matrices = gram_matrices
        self.C = C
        self.k_idx = k_idx

    def fit(self, X, y):
        self.train_idx_ = np.asarray(X, dtype=int).ravel()
        K_sub = self.gram_matrices[self.k_idx][np.ix_(self.train_idx_, self.train_idx_)]
        self.clf_ = SVC(kernel="precomputed", C=self.C, max_iter=20000).fit(K_sub, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        test_idx = np.asarray(X, dtype=int).ravel()
        return self.clf_.predict(self.gram_matrices[self.k_idx][np.ix_(test_idx, self.train_idx_)])


def nested_cv_evaluate(raw_grams, labels, n_outer=10, n_inner=5):
    grams = [K for K_raw in raw_grams for K in (K_raw, normalize_gram(K_raw))]
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED)

    accs = []
    for train_idx, test_idx in outer_cv.split(np.zeros(len(labels)), labels):
        search = GridSearchCV(
            PrecomputedKernelSVC(grams),
            param_grid={"C": C_VALUES, "k_idx": list(range(len(grams)))},
            cv=inner_cv,
        )
        search.fit(train_idx, labels[train_idx])
        accs.append(accuracy_score(labels[test_idx], search.predict(test_idx)))
    return float(np.mean(accs)) * 100.0, float(np.std(accs)) * 100.0


# Experiments
def sanity_check_implementation():
    print("=== 1. Validation: efficient vs. naive Kmgh on small MUTAG graphs ===")
    ds = fetch_dataset("MUTAG", verbose=False)
    small = [G for G in (to_nx(g) for g in ds.data[:15]) if G.number_of_nodes() <= 15][:5]
    max_diff = max(
        abs(efficient_multiplicate_graph_hopper(small[i], small[j], discrete_kappa_builder(small[i], small[j]))
            - naive_multiplicate_graph_hopper(small[i], small[j], discrete_kappa_builder(small[i], small[j])))
        for i in range(len(small)) for j in range(i, len(small))
    )
    print(f"  Checked {len(small)*(len(small)+1)//2} pairs, max |eff - naive| = {max_diff:.2e}\n")


def run_dataset(name, is_continuous=False):
    ds_disc = fetch_dataset(name, prefer_attr_nodes=False, verbose=False)
    if is_continuous:
        ds_attr = fetch_dataset(name, prefer_attr_nodes=True, verbose=False)
        idx = select_indices(ds_attr.data, ds_attr.target, MAX_GRAPHS_MAP.get(name), max_nodes=25, min_per_class=10)
        data_attr = [ds_attr.data[i] for i in idx]
        data_disc = [ds_disc.data[i] for i in idx]
        labels = np.asarray(ds_attr.target)[idx]
        nx_graphs = [to_nx(ga, gd) for ga, gd in zip(data_attr, data_disc)]
        tag = "Continuous"
    else:
        idx = select_indices(ds_disc.data, ds_disc.target, MAX_GRAPHS_MAP.get(name))
        data_disc = [ds_disc.data[i] for i in idx]
        labels = np.asarray(ds_disc.target)[idx]
        nx_graphs = [to_nx(g) for g in data_disc]
        tag = "Discrete"

    avg_n = np.mean([g.number_of_nodes() for g in nx_graphs])
    print(f"=== {tag} Dataset: {name} (N={len(nx_graphs)}, avg |V|={avg_n:.1f}) ===")
    results = []

    def record(kname, grams, dt):
        m, s = nested_cv_evaluate(grams, labels)
        results.append((kname, m, s, dt))
        print(f"\r  {kname:18s}: {m:5.2f} +/- {s:4.2f}%   (Gram time: {dt:8.3f}s)")

    for label_prefix, k_arg in [("VertexHist", VertexHistogram(normalize=False)), ("ShortestPath", ShortestPath(normalize=False))]:
        print(f"  running {label_prefix}...", end="", flush=True)
        K, dt = compute_grakel_gram(k_arg, data_disc)
        record(label_prefix, [K], dt)

    print("  running WeisfeilerLehman...", end="", flush=True)
    wl_pairs = [compute_grakel_gram(WeisfeilerLehman(n_iter=h, base_graph_kernel=VertexHistogram, normalize=False), data_disc) for h in WL_H_VALUES]
    record("WeisfeilerLehman", [K for K, _ in wl_pairs], sum(dt for _, dt in wl_pairs))

    print("  running GraphHopper...", end="", flush=True)
    if is_continuous:
        gh_pairs = [compute_grakel_gram(GraphHopper(kernel_type="linear", normalize=False), data_attr)] + [
            compute_grakel_gram(GraphHopper(kernel_type=("gaussian", g), normalize=False), data_attr) for g in GAMMA_VALUES
        ]
        record("GraphHopper", [K for K, _ in gh_pairs], sum(dt for _, dt in gh_pairs))
        # for kmgh, there's tqdm for printing
        mgh_pairs = [compute_kmgh_gram(nx_graphs, gaussian_kappa_builder(g)) for g in GAMMA_VALUES]
        record("Kmgh (ours)", [K for K, _ in mgh_pairs], sum(dt for _, dt in mgh_pairs))
    else:
        K_gh, dt_gh = compute_grakel_gram(GraphHopper(kernel_type="linear", normalize=False), to_grakel_onehot(data_disc))
        record("GraphHopper", [K_gh], dt_gh)
        # for kmgh, there's tqdm for printing
        K_mgh, dt_mgh = compute_kmgh_gram(nx_graphs, discrete_kappa_builder)
        record("Kmgh (ours)", [K_mgh], dt_mgh)

    print()
    return results


def run_size_scaling_experiment():
    print("=== Running Time vs. Graph Size (|V|) ===")
    rng = np.random.default_rng(SEED)
    n_graphs = 10
    families = {
        #regular is regular
        "4-Regular (deg=4)": lambda n, s: nx.random_regular_graph(4, n, seed=s),
        #barabasi is more clustered random graphs
        "Barabasi-Albert (m=2, avg deg~4)": lambda n, s: nx.barabasi_albert_graph(n, 2, seed=s),
    }

    scaling_results = {}
    for fam_name, gen in families.items():
        scaling_results[fam_name] = {"sizes": SCALING_SIZES, "Kmgh": [], "SP": [], "GH": []}
        print(f"  Family: {fam_name}")
        for n in SCALING_SIZES:
            nx_list = []
            for k in range(n_graphs):
                G = gen(n, SEED + n * 100 + k)
                labs = {u: int(rng.integers(0, 2)) for u in G.nodes()}
                nx.set_node_attributes(G, labs, "label")
                nx.set_node_attributes(G, {u: [1.0, 0.0] if v == 0 else [0.0, 1.0] for u, v in labs.items()}, "onehot")
                nx_list.append(G)

            _, t_mgh = compute_kmgh_gram(nx_list, discrete_kappa_builder)
            _, t_sp = compute_grakel_gram(ShortestPath(normalize=False), list(graph_from_networkx(nx_list, node_labels_tag="label")))
            _, t_gh = compute_grakel_gram(GraphHopper(kernel_type="linear", normalize=False), list(graph_from_networkx(nx_list, node_labels_tag="onehot")))

            scaling_results[fam_name]["Kmgh"].append(t_mgh)
            scaling_results[fam_name]["SP"].append(t_sp)
            scaling_results[fam_name]["GH"].append(t_gh)
            print(f"    |V|={n:3d}: Kmgh={t_mgh:8.3f}s  SP={t_sp:8.3f}s  GH={t_gh:8.3f}s")
    return scaling_results



# Plotting & CSV saving
def save_outputs(all_acc_results, scaling_results):
    ds_names = list(all_acc_results.keys())
    fig, axes = plt.subplots(1, len(ds_names), figsize=(4.5 * len(ds_names), 4.5))
    for ax, ds_name in zip(np.atleast_1d(axes), ds_names):
        rows = all_acc_results[ds_name]
        x = np.arange(len(rows))
        ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("10-fold CV Accuracy (%)")
        ax.set_title(ds_name)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("accuracy_comparison.png", dpi=200)
    plt.close(fig)

    for use_log, fname in [(True, "runtime_vs_size_loglog.png"), (False, "runtime_vs_size.png")]:
        fig, axes = plt.subplots(1, len(scaling_results), figsize=(5.5 * len(scaling_results), 4.5))
        for ax, (fam_name, res) in zip(np.atleast_1d(axes), scaling_results.items()):
            plot_fn = ax.loglog if use_log else ax.plot
            plot_fn(res["sizes"], res["Kmgh"], "o-", label="Kmgh (ours)")
            plot_fn(res["sizes"], res["SP"], "s--", label="ShortestPath (GraKeL)")
            plot_fn(res["sizes"], res["GH"], "^:", label="GraphHopper (GraKeL)")
            ax.set_xlabel("Number of vertices |V|")
            ax.set_ylabel("10x10 Gram matrix time (s)")
            ax.set_title(fam_name)
            ax.legend(fontsize=8)
            ax.grid(True, which="both" if use_log else "major", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fname, dpi=200)
        plt.close(fig)

    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Kernel", "Mean_Accuracy_Percent", "Std_Accuracy_Percent", "Gram_Time_Seconds"])
        for ds_name in ds_names:
            for kname, m, s, dt in all_acc_results[ds_name]:
                w.writerow([ds_name, kname, f"{m:.2f}", f"{s:.2f}", f"{dt:.4f}"])

    with open("scaling_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Graph_Family", "Num_Vertices", "Kmgh_Time_Seconds", "ShortestPath_Time_Seconds", "GraphHopper_Time_Seconds"])
        for fam_name, res in scaling_results.items():
            for idx, n in enumerate(res["sizes"]):
                w.writerow([fam_name, n, f"{res['Kmgh'][idx]:.4f}", f"{res['SP'][idx]:.4f}", f"{res['GH'][idx]:.4f}"])
    print("Saved plots and CSVs.")


def experimentation():
    warnings.filterwarnings("ignore") # could otherwise brick tqdm unnecessarily
    sanity_check_implementation()

    all_acc_results = {
        "MUTAG": run_dataset("MUTAG", is_continuous=False),
        "PTC_FM": run_dataset("PTC_FM", is_continuous=False),
        "ENZYMES": run_dataset("ENZYMES", is_continuous=True),
        "PROTEINS_full": run_dataset("PROTEINS_full", is_continuous=True),
    }
    scaling_results = run_size_scaling_experiment()
    save_outputs(all_acc_results, scaling_results)


if __name__ == "__main__":
    experimentation()