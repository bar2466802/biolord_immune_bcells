import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
import utils.settings as settings
from collections import Counter
from random import choice



def get_labels_as_array(row, property_):
    return np.array(list(row[property_].replace('[', '').replace(']', '').strip().split()), dtype=int)

def get_counters(ground_truth_labels, labels, num_unique_true_labels):
    label_counts = {label: Counter([ground_truth_labels[i] for i in range(len(labels)) if labels[i] == label]) for label in set(labels)}
    label_counts_df = pd.DataFrame(label_counts).sort_index()
    label_counts_df /= num_unique_true_labels
    new_labels = [label_counts[label].most_common(1)[0][0] for label in labels]

    return label_counts_df, new_labels
    # property_ = 'kmeans_' + attribute
    # df[property_] = new_labels
    # df[property_] = df[property_].replace(attributes_map[attribute])


def filter_df(df):
    return df.loc[(df['id_biolord'] == 114) & (df['score_name'] == 'Adjusted Mutual Information')]


def get_random_max_index(labels_list):
    max_value = np.nanmax(labels_list)
    # Get a list of indices where the maximum value occurs
    max_indices = [i for i, x in enumerate(labels_list) if x == max_value]
    # Get a random index from the list of indices
    random_index = choice(max_indices)
    return random_index


def robustness(df_celltype_path, df_organ_path):
    df_celltype = pd.read_csv(df_celltype_path)
    df_organ = pd.read_csv(df_organ_path)

    df_celltype = filter_df(df_celltype)
    df_organ = filter_df(df_organ)

    values = ['B1', 'CYCLING_B', 'IMMATURE_B', 'LARGE_PRE_B', 'LATE_PRO_B', 'MATURE_B', 'PLASMA_B', 'PRE_PRO_B', 'PRO_B',
              'SMALL_PRE_B', 'BM', 'GU', 'KI', 'LI', 'MLN', 'SK', 'SP', 'TH', 'YS']
    labels_dic = {i: value for i, value in enumerate(values)}

    for (n_clusters_celltype, group_celltype), (n_clusters_organ, group_organ) in zip(df_celltype.groupby(['n_clusters']), df_organ.groupby(['n_clusters'])):
        new_labels_list = []
        df_robustness = {
            'attribute': [],
            'value': [],
            'k': [],
            'n_clusters': []
        }
        for (index_celltype, row_celltype), (index_organ, row_organ) in zip(group_celltype.iterrows(), group_organ.iterrows()):
            true_labels_celltype = get_labels_as_array(row_celltype, 'true_labels')
            labels_celltype = get_labels_as_array(row_celltype, 'labels')
            true_labels_organ = get_labels_as_array(row_organ, 'true_labels')
            labels_organ = get_labels_as_array(row_organ, 'labels')
            number_of_celltypes = len(set(true_labels_celltype))
            number_of_organs = len(set(true_labels_organ))

            label_counts_celltype, new_labels_celltype = get_counters(true_labels_celltype, labels_celltype, number_of_organs)
            label_counts_organ, new_labels_organ = get_counters(true_labels_organ, labels_organ, number_of_celltypes)
            new_labels = []
            for k in list(set(labels_celltype)):
                if label_counts_celltype[k].max() > label_counts_organ[k].max():
                    new_val = get_random_max_index(label_counts_celltype[k])
                    max_val = label_counts_celltype[k].max()
                    new_labels.append(new_val)
                else:
                    new_val = number_of_celltypes + get_random_max_index(label_counts_organ[k])
                    max_val = label_counts_organ[k].max()
                    new_labels.append(new_val)

                df_robustness['attribute'].append(new_val)
                df_robustness['value'].append(max_val)
                df_robustness['n_clusters'].append(n_clusters_celltype)
                df_robustness['k'].append(k)
            new_labels_list.append(new_labels)

        most_common_new_labels = [Counter(col).most_common(1)[0][0] for col in zip(*new_labels_list)]
        meaningful_new_labels = [labels_dic.get(item, item) for item in most_common_new_labels]
        print(f'for n_clusters = {n_clusters_celltype}\nmost_common_new_labels = {meaningful_new_labels}')
        df_robustness = pd.DataFrame(df_robustness)
        df_robustness['attribute'] = df_robustness['attribute'].replace(labels_dic)

        df_robustness = df_robustness.drop_duplicates()

        for idx, val in enumerate(meaningful_new_labels):
            mask = (df_robustness.k == idx) & (df_robustness.attribute != val)
            df_robustness = df_robustness.loc[~mask]
        df_robustness = df_robustness.reset_index()
        print(df_robustness)

    # for row1, row2 in zip(df_celltype.iterrows(), df_organ.iterrows()):
    #     index_celltype, row_celltype = row1
    #     index_organ, row_organ = row2
    #     true_labels_celltype = get_labels_as_array(row_celltype, 'true_labels')
    #     labels_celltype = get_labels_as_array(row_celltype, 'labels')
    #     true_labels_organ = get_labels_as_array(row_organ, 'true_labels')
    #     labels_organ = get_labels_as_array(row_organ, 'labels')
    #
    #     tuples_celltype_organ = list(zip(true_labels_celltype, true_labels_organ))
    #
    #     label_counts_celltype, new_labels_celltype = get_counters(true_labels_celltype, labels_celltype)
    #     label_counts_organ, new_labels_organ = get_counters(true_labels_organ, labels_organ)
    #
    #     number_of_celltypes = len(set(true_labels_celltype))
    #     number_of_organs = len(set(true_labels_organ))
    #     new_labels = []
    #     for k in list(set(labels_celltype)):
    #         if label_counts_celltype[k].max() > label_counts_organ[k].max():
    #             new_labels.append(label_counts_celltype[k].argmax())
    #         else:
    #             new_labels.append(number_of_celltypes + label_counts_organ[k].argmax())
    #
    #     print(new_labels)
        # organ_together_count = []
        # for organ in set(true_labels_organ):
        #     organ_index = true_labels_organ == organ
        #     k_means_labels_of_organ = labels_celltype[organ_index]
        #     labels_count_dic = dict(Counter(k_means_labels_of_organ))
        #
        #     a = 1

    print("this is the end")


if __name__ == "__main__":
    settings.init()
    df_celltype_path = "../output/celltype_kmeans_models_scores.csv"
    df_organ_path = "../output/organ_kmeans_models_scores.csv"
    robustness(df_celltype_path, df_organ_path)
