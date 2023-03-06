################################################
# FILE: cluster_analysis.py
# WRITERS: Bar Melinarskiy
# DESCRIPTION: Utils functions for evaluating the received latent space clusters
################################################

# Imports
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import rand_score


# Functions
def rand_index(df: pd.DataFrame, k: int, ground_truth_labels: np.ndarray):
    """
    Evaluate the latent space clustering, comparing known labels to kmeans clustering using the Rand index score
    :param df: The latent space data
    :param k: Number of clusters - usually the number of labels (cell types / organs / etc...)
    :param ground_truth_labels: array of known labels for each sample
    :return: rand index score
    """
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(df)
    return rand_score(ground_truth_labels, labels)
