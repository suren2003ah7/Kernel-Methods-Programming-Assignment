import os
from re import search

import matplotlib.pyplot as plt
import numpy as np

from grakel.datasets import fetch_dataset
from grakel.kernels import VertexHistogram, WeisfeilerLehman
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.svm import SVC


dataset = fetch_dataset("MUTAG", verbose=False)
graphs = dataset.data
labels = np.asarray(dataset.target)

C_VALUES = 10.0 ** np.arange(-3, 4)
H_VALUES = range(6)


class PrecomputedKernelSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, gram_matrices, C=1.0, h=0):
        self.gram_matrices = gram_matrices
        self.C = C
        self.h = h

    def fit(self, X, y):
        self.train_indices_ = np.asarray(X, dtype=int).reshape(-1)
        train_kernel = self.gram_matrices[self.h][np.ix_(self.train_indices_, self.train_indices_)] # Select submatrix for current inner fold
        self.classifier_ = SVC(kernel="precomputed", C=self.C).fit(train_kernel, y)
        self.classes_ = self.classifier_.classes_
        return self

    def predict(self, X):
        test_indices = np.asarray(X, dtype=int).reshape(-1)
        test_kernel = self.gram_matrices[self.h][np.ix_(test_indices, self.train_indices_)]
        return self.classifier_.predict(test_kernel)

    def predict_with_kernel(self, test_kernel):
        return self.classifier_.predict(test_kernel)


def wl_kernel_svm(graphs, labels, c_values=None, h=5):
    if c_values is None:
        c_values = C_VALUES

    # Splits the dataset into a training and a test set
    G_train, G_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.1, random_state=42)

    if h == 0:
        gk = VertexHistogram(normalize=True)
    else:
        gk = WeisfeilerLehman(
            n_iter=h,
            base_graph_kernel=VertexHistogram,
            normalize=True,
        )

    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)

    accs = []
    train_accs = []
    for c in c_values:

        clf = SVC(kernel="precomputed", C=c)
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        train_acc = accuracy_score(y_train, clf.predict(K_train))
        train_accs.append(train_acc)

        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)

    return accs, train_accs


def compute_wl_gram_matrices(graphs, h_values=H_VALUES):
    gram_matrices = {}
    for h in h_values:
        if h == 0:
            kernel = VertexHistogram(normalize=True)
        else:
            kernel = WeisfeilerLehman(
                n_iter=h,
                base_graph_kernel=VertexHistogram,
                normalize=True,
            )
        gram_matrices[h] = kernel.fit_transform(graphs)
    return gram_matrices


def wl_with_joint_grid_search_cross_val(graphs, labels, full_gram_matrices):
    c_values = C_VALUES
    h_values = H_VALUES

    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=43)

    outer_predictions = np.empty_like(labels)
    fold_accuracies = []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(len(labels)), labels), start=1
    ):
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        gram_matrices = {   # Select fold matrices from global gram matrices
            h: full_gram_matrices[h][np.ix_(train_idx, train_idx)]
            for h in h_values
        }

        search = GridSearchCV(
            PrecomputedKernelSVC(gram_matrices=gram_matrices),
            param_grid={"C": c_values, "h": list(h_values)},
            scoring="accuracy",
            cv=inner_cv,
        )
        search.fit(np.arange(len(train_idx)), y_train)

        best_h = search.best_params_["h"]
        best_c = search.best_params_["C"]
        test_kernel = full_gram_matrices[best_h][np.ix_(test_idx, train_idx)]   # Select Test kernel for best h from global gram
        y_pred = search.best_estimator_.predict_with_kernel(test_kernel)
        outer_predictions[test_idx] = y_pred

        train_kernel = gram_matrices[best_h]
        y_pred_train = search.best_estimator_.predict_with_kernel(train_kernel)
        train_accuracy = accuracy_score(y_train, y_pred_train)

        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_accuracy)
        fold_results.append(
            {
                "fold": fold,
                "h": best_h,
                "C": best_c,
                "accuracy": fold_accuracy,
                "train_accuracy": train_accuracy,
            }
        )
        print(
            f"Joint GridSearch fold {fold}: accuracy={fold_accuracy:.4f}, "
            f"h={best_h}, C={best_c:g}"
        )

    pooled_accuracy = accuracy_score(labels, outer_predictions)
    mean_fold_accuracy = np.mean(fold_accuracies)
    print(f"Joint GridSearch mean outer-fold accuracy: {mean_fold_accuracy * 100:.2f}%")
    print(f"Joint GridSearch pooled outer-CV accuracy: {pooled_accuracy * 100:.2f}%")
    return {
        "fold_results": fold_results,
        "mean_fold_accuracy": mean_fold_accuracy,
        "pooled_accuracy": pooled_accuracy,
    }


gram_matrices = compute_wl_gram_matrices(graphs)

results = []
for h in range(6):
    print(f"Running WL kernel SVM with h={h}")
    results.append(wl_kernel_svm(graphs, labels, h=h))

res = wl_with_joint_grid_search_cross_val(graphs, labels, gram_matrices)


## Plot results

comparison_fig, (comparison_test_ax, comparison_train_ax) = plt.subplots(
    2, 1, sharex=True, figsize=(10, 9)
)
h_colors = plt.get_cmap("tab10")
for h, (accuracies, train_accuracies) in enumerate(results):
    color = h_colors(h)
    test_accuracies_percent = np.asarray(accuracies) * 100
    train_accuracies_percent = np.asarray(train_accuracies) * 100

    comparison_test_ax.plot(
        C_VALUES,
        test_accuracies_percent,
        marker="o",
        color=color,
        label=f"h={h}",
    )
    comparison_train_ax.plot(
        C_VALUES,
        train_accuracies_percent,
        marker="o",
        color=color,
        label=f"h={h}",
    )

cv_h_seen = set()
for fold_result in res["fold_results"]:
    h = fold_result["h"]
    label = f"h={h} outer CV" if h not in cv_h_seen else None
    cv_h_seen.add(h)
    accuracy_percent = fold_result["accuracy"] * 100
    comparison_test_ax.scatter(
        fold_result["C"],
        accuracy_percent,
        marker="X",
        s=75,
        color=h_colors(h),
        edgecolors="black",
        linewidths=0.5,
        label=label,
        zorder=3,
    )
    comparison_test_ax.annotate(
        f"F{fold_result['fold']}",
        (fold_result["C"], accuracy_percent),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
    )

cv_h_seen = set()
for fold_result in res["fold_results"]:
    h = fold_result["h"]
    label = f"h={h} outer CV" if h not in cv_h_seen else None
    cv_h_seen.add(h)
    accuracy_percent = fold_result["train_accuracy"] * 100
    comparison_train_ax.scatter(
        fold_result["C"],
        accuracy_percent,
        marker="X",
        s=75,
        color=h_colors(h),
        edgecolors="black",
        linewidths=0.5,
        label=label,
        zorder=3,
    )
    comparison_train_ax.annotate(
        f"F{fold_result['fold']}",
        (fold_result["C"], accuracy_percent),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
    )

comparison_test_ax.axhline(
    res["pooled_accuracy"] * 100,
    color="black",
    linestyle=":",
    label=f"mean outer accuracy: {res['mean_fold_accuracy'] * 100:.1f}%",
)

for ax in (comparison_test_ax, comparison_train_ax):
    ax.set_xscale("log")
    ax.set_xticks(C_VALUES, [f"{c:g}" for c in C_VALUES])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 105)    # To see the 100% still in the graph
    ax.grid(True, alpha=0.3)
    ax.legend(title="h and evaluation", fontsize=8, ncol=2)

comparison_test_ax.set_title("test curves and optimization results")
comparison_train_ax.set_title("training curves and optimization results")
comparison_train_ax.set_xlabel("SVM regularization parameter C")
comparison_fig.suptitle("WL SVM hyperparameter-optimization comparison")
comparison_fig.tight_layout(rect=(0, 0, 1, 0.96))
comparison_fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"wl_hyperparam_cv_comparison.png"), dpi=300)
