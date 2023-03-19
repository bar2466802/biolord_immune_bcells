################################################
# FILE: cluster_analysis.py
# WRITERS: Bar Melinarskiy
# DESCRIPTION: Utils functions for evaluating the received latent space clusters
################################################

# Imports
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
import seaborn as sns
from math import ceil
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import rand_score
from yellowbrick.cluster import SilhouetteVisualizer  # to continue here
from collections import Counter
from formatters import *


rng = np.random.RandomState(0)

"""
* V-measure, the harmonic mean of completeness and homogeneity;

* Rand index, which measures how frequently pairs of data points are grouped consistently according to the result 
of the clustering algorithm and the ground truth class assignment;

* Adjusted Rand index (ARI), a chance-adjusted Rand index such that a random cluster assignment has an
ARI of 0.0 in expectation;

* Mutual Information (MI) is an information theoretic measure that quantifies how dependent are the two
labelings. Note that the maximum value of MI for perfect labelings depends on the number of clusters and samples;

* Normalized Mutual Information (NMI), a Mutual Information defined between 0 (no mutual information) in the
limit of large number of data points and 1 (perfectly matching label assignments, up to a permutation of the labels). 
It is not adjusted for chance: then the number of clustered data points is not large enough, the expected values of MI
or NMI for random labelings can be significantly non-zero;

* Adjusted Mutual Information (AMI), a chance-adjusted Mutual Information. Similarly to ARI, random cluster
assignment has an AMI of 0.0 in expectation.
"""
score_funcs = [
    ("Homogeneity", metrics.homogeneity_score),
    ("Completeness", metrics.completeness_score),
    ("V-measure", metrics.v_measure_score),
    ("Rand index", metrics.rand_score),
    ("Adjusted Rand Index", metrics.adjusted_rand_score),
    ("MI", metrics.mutual_info_score),
    ("NMI", metrics.normalized_mutual_info_score),
    ("Adjusted Mutual Information", metrics.adjusted_mutual_info_score),
]

ATTRIBUTES = ['celltype', 'organ']


# Functions
def rand_index(X: np.ndarray, true_labels: np.ndarray, df: pd.DataFrame, attributes_map, title: str, save_path: str) -> np.ndarray:
    """
    Evaluate the latent space clustering, comparing known labels to kmeans clustering using the Rand index score
    :param X: The latent space data
    :param true_labels: array of known labels for each sample
    :param df: dataframe for UMAP
    :param title: title for plot
    :param save_path: path to save the figures
    :return: rand index score for all attributes together and separately
    """
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(true_labels)) - (1 if -1 in true_labels else 0)
    n_noise_ = list(true_labels).count(-1)
    k_values = np.arange(2, 16)
    df_metrics = {
        "k": k_values,
        "Homogeneity": [],
        "Completeness": [],
        "V-measure": [],
        "Homogeneity": [],
        "Adjusted Rand Index": [],
        "Adjusted Mutual Information": [],
        "Silhouette Coefficient": []
    }

    fixed_classes_uniform_labelings_scores_plot(X, true_labels, title, save_path)
    uniform_labelings_scores_plot(X, true_labels, title, save_path)
    best_kmeans = kmeans_scores_plot(X, true_labels, title, save_path)
    # print("kmeans labels are:", best_kmeans['labels'])
    umap_with_kmeans_labels(df, best_kmeans, title, save_path, attributes_map)
    return df_metrics


def random_labels(n_samples, n_classes):
    return rng.randint(low=0, high=n_classes, size=n_samples)


def fixed_classes_uniform_labelings_scores(
        score_func, n_samples, n_clusters_range, n_classes, n_runs=5
):
    scores = np.zeros((len(n_clusters_range), n_runs))
    labels_a = random_labels(n_samples=n_samples, n_classes=n_classes)

    for i, n_clusters in enumerate(n_clusters_range):
        for j in range(n_runs):
            labels_b = random_labels(n_samples=n_samples, n_classes=n_clusters)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


def fixed_classes_uniform_labelings_scores_plot(X, true_labels, title, save_path):
    n_samples = len(true_labels)
    n_classes = len(set(true_labels)) - (1 if -1 in true_labels else 0)
    n_clusters_range = np.arange(2, 16).astype(int)
    plots = []
    names = []

    sns.color_palette("colorblind")
    plt.figure(1)

    for marker, (score_name, score_func) in zip("d^vx.+*,", score_funcs):
        scores = fixed_classes_uniform_labelings_scores(
            score_func, n_samples, n_clusters_range, n_classes=n_classes
        )
        plots.append(
            plt.errorbar(
                n_clusters_range,
                scores.mean(axis=1),
                scores.std(axis=1),
                alpha=0.8,
                linewidth=1,
                marker=marker,
            )[0]
        )
        names.append(score_name)
    plt.title(title +
              ", Clustering measures for random uniform labeling\n"
              f"against reference assignment with {n_classes} classes"
              )
    plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")
    plt.ylabel("Score value")
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(plots, names, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(save_path + "fixed_classes_uniform_labelings_scores.png", format="png", dpi=300)
    plt.show()


def uniform_labelings_scores(score_func, n_samples, n_clusters_range, n_runs=5):
    scores = np.zeros((len(n_clusters_range), n_runs))

    for i, n_clusters in enumerate(n_clusters_range):
        for j in range(n_runs):
            labels_a = random_labels(n_samples=n_samples, n_classes=n_clusters)
            labels_b = random_labels(n_samples=n_samples, n_classes=n_clusters)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


def uniform_labelings_scores_plot(X, true_labels, title, save_path):
    n_samples = len(true_labels)
    n_clusters_range = np.arange(2, 16).astype(int)
    plt.figure(2)
    plots = []
    names = []

    for marker, (score_name, score_func) in zip("d^vx.+*", score_funcs):
        scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
        plots.append(
            plt.errorbar(
                n_clusters_range,
                np.median(scores, axis=1),
                scores.std(axis=1),
                alpha=0.8,
                linewidth=2,
                marker=marker,
            )[0]
        )
        names.append(score_name)

    plt.title(title +
              ", Clustering measures for 2 random uniform labelings\nwith equal number of clusters"
              )
    plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")
    plt.ylabel("Score value")
    plt.legend(plots, names, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(bottom=-0.05, top=1.05)
    plt.savefig(save_path + "uniform_labelings_scores.png", format="png", dpi=300)
    plt.show()


def kmeans_scores(X, true_labels, score_func, n_clusters_range, n_runs=5):
    scores = np.zeros((len(n_clusters_range), n_runs))
    best_kmeans = {
        "labels": [],
        "n_clusters": 0,
        "km": None
    }
    max_score = -np.inf
    for i, n_clusters in enumerate(n_clusters_range):
        for j in range(n_runs):
            km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
            labels_kmeans = km.fit_predict(X)
            score = score_func(true_labels, labels_kmeans)
            if score > max_score:
                max_score = score
                best_kmeans['labels'] = labels_kmeans
                best_kmeans['n_clusters'] = n_clusters
                best_kmeans['km'] = km
            scores[i, j] = score
    best_kmeans['score'] = max_score
    return scores, best_kmeans


def kmeans_scores_plot(X, true_labels, title, save_path):
    n_samples = len(true_labels)
    n_clusters_range = np.arange(2, 16).astype(int)
    plt.figure(3)
    plots = []
    names = []
    ari_best_kmeans = None
    for marker, (score_name, score_func) in zip("d^vx.+*", score_funcs):
        scores, best_kmeans = kmeans_scores(X, true_labels, score_func, n_clusters_range)
        if score_func == metrics.adjusted_rand_score:
            ari_best_kmeans = best_kmeans
        plots.append(
            plt.errorbar(
                n_clusters_range,
                np.median(scores, axis=1),
                scores.std(axis=1),
                alpha=0.8,
                linewidth=2,
                marker=marker,
            )[0]
        )
        names.append(score_name)

    plt.title(title +
              ", Clustering measures for Kmeans Vs ground trouth labels"
              )
    plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")
    plt.ylabel("Score value")
    plt.legend(plots, names, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(bottom=-0.05, top=1.05)
    plt.tight_layout()
    plt.savefig(save_path + "kmeans_scores_plot.png", format="png", dpi=300)
    plt.show()
    return ari_best_kmeans


def umap_with_kmeans_labels(df, best_kmeans, title, save_path, attributes_map):
    fig, axs = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={"width_ratios": [1, 1, 1]})
    # plt.subplots_adjust(wspace=0.2)
    # df['kmeans'] = best_kmeans['labels']
    title += '\nBest kmeans got Adjusted Rand Index score of: ' + str(round(best_kmeans['score'], 3)) + \
             ' with n_clusters = ' + str(best_kmeans['n_clusters'])

    for attribute in ATTRIBUTES:
        # Determine the most common label for each label in best_kmeans['labels']
        # print("kmeans values before are:", best_kmeans['labels'], '\n')
        attribute_key = attribute + '_key'
        ground_truth_labels = df[attribute_key]
        label_counts = {label: Counter([ground_truth_labels[i] for i in range(len(best_kmeans['labels'])) if best_kmeans['labels'][i] == label]) for label in
                        set(best_kmeans['labels'])}
        new_labels = [label_counts[label].most_common(1)[0][0] for label in best_kmeans['labels']]
        property_ = 'kmeans_' + attribute
        df[property_] = new_labels
        # print(f'att:{attribute}')
        # print(attributes_map[attribute])
        # print(f'kmeans: {property_}')
        # print(df[property_])
        df[property_] = df[property_].replace(attributes_map[attribute])
        # print(print(f'kmeans: {property_} after replace:'))
        # print(df[property_])
        if attribute == "celltype":
            df[property_] = switch_to_celltype_fullname(df[property_])
        elif attribute == "organ":
            df[property_] = switch_to_organ_fullname(df[property_])
        # print(print(f'kmeans: {property_} after formmating:'))
        # print(df[property_])
        # print('-------------------------------------------')

        labels_truth = df['celltype_key']
    # unique_labels_kmeans = list(set(best_kmeans['labels']))
    # for l in unique_labels_kmeans:
    #     indexes = [i for i in range(len(best_kmeans['labels'])) if best_kmeans['labels'][i] == l]
    #     values = [labels_truth[i] for i in indexes]
    #     counts = Counter(values)
    #     total_count = len(values)
    #     print(f'For the label {l}:')
    #     for value, count in counts.items():
    #         percentage = count / total_count * 100
    #         print(f'{value}: {percentage}%')
    #     most_common_label = max(counts, key=counts.get)
    #     print(f'For the label {l} the new label will be: {most_common_label}')
    #
    # print("kmeans values after are:", new_labels, '\n')
    # print("indexes of ground truth:")
    # map_indexes_truth = print_indexes(list(df['celltype_key']))
    # print("----------------------------------------")
    # print("indexes of kmeans before:")
    # print("----------------------------------------")
    # map_indexes_kmeans = pd.DataFrame(print_indexes(list(best_kmeans['labels'])), index=[0])
    # print("map before:")
    # print(map_indexes_kmeans)
    # map_indexes_kmeans.replace(map_indexes_truth, inplace=True)
    # print("map after:")
    # print(map_indexes_kmeans)
    #
    # print("indexes of kmeans after:")
    # print_indexes(new_labels)
    # print("----------------------------------------")

    # 2nd attempted
    # Determine the majority label in labels1
    # majority_label = Counter(labels1).most_common(1)[0][0]
    # # Create a dictionary mapping each label in labels1 to its corresponding ID
    # id_dict = {label: i for i, label in enumerate(set(labels1))}
    # # Transfer the IDs from labels1 to labels2 based on majority rule
    # ids2 = [id_dict[label] if label == majority_label else -1 for label in labels2]

    for col, hue_attribute in enumerate(['organ', 'celltype', 'kmeans_celltype']):
        sns.scatterplot(
            data=df,
            x="umap1",
            y="umap2",
            hue=hue_attribute,
            ax=axs[col],
            alpha=.8,
            s=60,
            palette="deep"
        )
        unique_labels = list(set(df[hue_attribute]))
        subplot_title = "Color is set to: " + hue_attribute + ", # of unique labels: " + str(unique_labels)
        axs[col].set_title(subplot_title)
        axs[col].set(xticklabels=[], yticklabels=[])
        axs[col].set_xlabel("UMAP1")
        axs[col].set_ylabel("UMAP2")
        axs[col].grid(False)
        axs[col].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    fig.tight_layout(pad=5.0)
    fig.suptitle(title, fontsize=14)
    plt.savefig(save_path + "umap_with_kmeans_labels.png", format="png", dpi=300)
    plt.show()


def print_indexes(list_):
    map_indexes = {}
    for value in set(list_):
        indexes = [i for i, x in enumerate(list_) if x == value]
        print(f"Indexes of {value}: {indexes}")
        for i in indexes:
            map_indexes[str(i)] = value
    return map_indexes


def silhouette(km, X):
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
    visualizer.fit(X)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure
