import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
import utils.settings as settings
from collections import Counter
from random import choice
from plotly.subplots import make_subplots
import colorcet as cc
import plotly.graph_objects as go
import plotly.express as px
import math
import os
import scanpy as sc
import umap.plot

import sys
sys.path.append("/cs/usr/bar246802/bar246802/SandBox2023/biolord_immune_bcells/utils")  # add utils
from utils.formatters import *


def get_labels_as_array(row, property_):
    return np.array(list(row[property_].replace('[', '').replace(']', '').strip().split()), dtype=int)


def get_counters(ground_truth_labels, labels, num_unique_true_labels):
    label_counts = {label: Counter([ground_truth_labels[i] for i in range(len(labels)) if labels[i] == label]) for label
                    in set(labels)}
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

    values = ['B1', 'CYCLING_B', 'IMMATURE_B', 'LARGE_PRE_B', 'LATE_PRO_B', 'MATURE_B', 'PLASMA_B', 'PRE_PRO_B',
              'PRO_B',
              'SMALL_PRE_B', 'BM', 'GU', 'KI', 'LI', 'MLN', 'SK', 'SP', 'TH', 'YS']
    labels_dic = {i: value for i, value in enumerate(values)}

    for (n_clusters_celltype, group_celltype), (n_clusters_organ, group_organ) in zip(
            df_celltype.groupby(['n_clusters']), df_organ.groupby(['n_clusters'])):
        new_labels_list = []
        df_robustness = {
            'attribute': [],
            'value': [],
            'k': [],
            'n_clusters': []
        }
        for (index_celltype, row_celltype), (index_organ, row_organ) in zip(group_celltype.iterrows(),
                                                                            group_organ.iterrows()):
            true_labels_celltype = get_labels_as_array(row_celltype, 'true_labels')
            labels_celltype = get_labels_as_array(row_celltype, 'labels')
            true_labels_organ = get_labels_as_array(row_organ, 'true_labels')
            labels_organ = get_labels_as_array(row_organ, 'labels')
            number_of_celltypes = len(set(true_labels_celltype))
            number_of_organs = len(set(true_labels_organ))

            label_counts_celltype, new_labels_celltype = get_counters(true_labels_celltype, labels_celltype,
                                                                      number_of_organs)
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


def plot_robustness(path="../output/2/trained_model_scores.csv", save_png_path="biolord_b_cells_scores.png"):
    df_scores = pd.read_csv(path)
    models_ids = list(set(df_scores['id_biolord']))
    print(f"There are : {len(df_scores)} lines in the scores csv, with {len(models_ids)} unique repeats")

    # scores_cols = [col for col in df_scores.columns if 'score' in col]
    scores_cols = ['score_robustness_mean', 'score_robustness_std', 'score_V-measure_celltype', 'score_V-measure_organ']
    #                'score_Adjusted Rand Index_celltype', 'score_Adjusted Rand Index_organ']

    # scores_cols = ['score_robustness_mean', 'score_robustness_std']
    palette = sns.color_palette(cc.glasbey, len(scores_cols)).as_hex()
    scores_color_map = dict(zip(np.unique(scores_cols), palette))

    arr_n_clusters = list(set(df_scores['n_clusters']))
    rows = math.ceil(len(arr_n_clusters) / 2)
    subplot_titles = [f"n_clusters = {n_clusters}" for n_clusters in arr_n_clusters]
    fig = make_subplots(rows, 2, subplot_titles=subplot_titles, vertical_spacing=0.14)

    row = col = 1
    for n_clusters, group in df_scores.groupby(['n_clusters']):
        for score_name in scores_cols:
            fig.append_trace(go.Scatter(
                x=group['id_biolord'],
                y=group[score_name],
                legendgroup=score_name,
                name=score_name,
                mode='lines',
                line=dict(color=scores_color_map[score_name], width=1)
            ), row=row, col=col)
            if col != 1 or row != 1:
                fig.update_traces(showlegend=False, row=row, col=col)
            # axs[row][col].plot(group['id_biolord'], group[score_name], color=scores_color_map[score_name],
            #                  label=score_name)
        # subplot_title = f"scores of model, n_clusters = {n_clusters}"
        # axs[row][col].set_title(subplot_title)
        fig.update_yaxes(range=[0, 1], title_text="Score", row=row, col=col)
        fig.update_xaxes(title_text="Id of model")
        # axs[row][col].set_xlabel("Id of model")
        # axs[row][col].set_ylabel("Score")
        # axs[row][col].grid(False)
        # axs[row][col].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        col += 1
        if col == 3:
            row += 1
            col = 1
    title = "Scores of trained biolord model grouped by k means n_clusters"
    # fig.tight_layout(pad=5.0)
    fig.update_layout(height=800, width=1250, title_text=title)
    # fig.suptitle(title, fontsize=14)
    # plt.savefig(save_png_path, format="png", dpi=300)
    fig.show()
    # fig.write_image(save_png_path)


def create_csv_new_labels(path="../output/2"):
    # Loop through all files in the directory
    df_all_labels = None
    for filename in os.listdir(path):
        # Check if the file starts with 'relabling'
        if 'relabeling' in filename:
            df_labels = pd.read_csv(path + '/' + filename)
            df_labels = df_labels.drop('Unnamed: 0', axis=1)
            df_labels['biolord_id'] = filename.split('_')[1].replace('.csv', '')
            if df_all_labels is not None:
                df_all_labels = pd.concat([df_all_labels, df_labels], ignore_index=True)
            else:
                df_all_labels = df_labels

    df_all_labels = df_all_labels.sort_values(by=['biolord_id', 'n_clusters'], ascending=True)
    df_all_labels = df_all_labels[['biolord_id', 'n_clusters', 'k', 'new_id', 'value']]
    df_all_labels.to_csv(settings.SAVE_DIR + "all_new_labels.csv", index=False)


# def count_new_labels(path="../output/all_new_labels.csv"):
#     df_all_labels = pd.read_csv(path)
#     all_new_id_counts = df_all_labels['new_id'].value_counts(normalize=True)
#     # print(f'for all df the new ids count is:')
#     # print(all_new_id_counts)
#
#     df = None
#
#     for n_clusters, group in df_all_labels.groupby('n_clusters'):
#         # print('******************************************************')
#         group_new_id_counts = group['new_id'].value_counts(normalize=True)
#         # print(f'for cluster {n_clusters} :')
#         # print(group_new_id_counts)
#
#         robustness_means = []
#         for id_ in group_new_id_counts.index:
#             new_id_value_mean = group[group['new_id'] == id_]['value'].mean()
#             robustness_means.append(new_id_value_mean)
#
#         percentages = np.array(group_new_id_counts.values) * 100
#         # values = np.concatenate((robustness_means, percentages))
#         # value_type = np.concatenate((['robustness_means'] * len(robustness_means),
#         #                              ['percentage'] * len(group_new_id_counts.values)))
#
#         # df_group_new_id_counts = pd.DataFrame({'new_id': np.repeat(group_new_id_counts.index, 2),
#         #                                        'value': values,
#         #                                        'value_type': value_type})
#         df_group_new_id_counts = pd.DataFrame({'new_id': group_new_id_counts.index,
#                                                'percentage': percentages,
#                                                'robustness_means': robustness_means})
#         df_group_new_id_counts['n_clusters'] = n_clusters
#
#         if df is not None:
#             df = pd.concat([df, df_group_new_id_counts], ignore_index=True)
#         else:
#             df = df_group_new_id_counts
#         # for k, group_k in group.groupby('k'):
#         #     group_new_id_counts = group_k['new_id'].value_counts(normalize=True)
#         #     print(f'for cluster {n_clusters} and k: {k} the new ids are:')
#         #     print(group_new_id_counts)
#         #
#         #     df_group_new_id_counts = pd.DataFrame({'new_id': group_new_id_counts.index, 'percentage': group_new_id_counts.values})
#         #     df_group_new_id_counts['n_clusters'] = n_clusters
#         #     df_group_new_id_counts['k'] = k
#         #
#         #     if df is not None:
#         #         df = pd.concat([df, df_group_new_id_counts], ignore_index=True)
#         #     else:
#         #         df = df_group_new_id_counts
#         # labels = list(set(group['new_id']))
#         # # count the frequency of each value
#         # counter = Counter(group['new_id'])
#         # # calculate the percentage of each value
#         # total = len(set(group['biolord_id']))
#         # percentage = {k: v / total * 100 for k, v in counter.items()}
#         # df_labels_percentage = pd.DataFrame(list(percentage.items()), columns=['new_id', 'percentage'])
#     # df['percentage'] *= 100
#     # df = df[['n_clusters', 'new_id', 'value_type', 'value']]
#     df = df[['n_clusters', 'new_id', 'robustness_means', 'percentage']]
#     # add columns for figure
#     celltypes = ['B1', 'CYCLING_B', 'IMMATURE_B', 'LARGE_PRE_B', 'LATE_PRO_B', 'MATURE_B', 'PLASMA_B', 'PRE_PRO_B',
#                  'PRO_B', 'SMALL_PRE_B']
#     organs = ['BM', 'GU', 'KI', 'LI', 'MLN', 'SK', 'SP', 'TH', 'YS']
#     df['id_type'] = df['new_id'].apply(lambda x: 'celltype' if x in celltypes else 'organ')
#
#     df = df[df['n_clusters'] > 8]
#     # print(df)
#     # # Group the DataFrame by group1 and group2, and get the top 3 max values for each group
#     # top3 = df.groupby(['n_clusters'])['percentage'].nlargest(3)
#     # # Get the indices of the top 3 max values
#     # indices = top3.index.get_level_values(level=1)
#     # # Filter the original DataFrame based on the indices of the top 3 max values
#     # result = df[df.index.isin(indices)]
#     # print(result)
#
#     df = df.sort_values(by=['n_clusters', 'id_type', "percentage", "robustness_means"], ascending=False)
#     df_celltype = df[df['id_type'] == 'celltype']
#     df_organ = df[df['id_type'] == 'organ']
#     # df['percentage'] = df['percentage'].astype(str)
#     # fig = px.bar(df, x="new_id", y="value", color="value_type", barmode="group",
#     #              facet_row="id_type", facet_col="n_clusters",
#     #              category_orders={"organ": organs,
#     #                               "celltype": celltypes})
#     # fig.show()
#
#     # create grouped bar chart
#     # fig = px.bar(df, x='new_id', y=['percentage', 'robustness_means'], color='new_id',
#     #              barmode='group', facet_col='n_clusters', facet_row="id_type")
#     # df_celltype = df[df['id_type'] == 'celltype']
#     # df['percentage'] = df['percentage'].astype(str)
#     # df['robustness_means'] = df['robustness_means'].astype(str)
#     # df_organ = df[df['id_type'] == 'organ']
#     # fig = go.Figure(
#     #     data=[
#     #         go.Bar(name='percentage', x=df_celltype['new_id'],
#     #                y=df_celltype['percentage'], yaxis='y',
#     #                offsetgroup=1),
#     #         go.Bar(name='robustness_means', x=df_celltype['new_id'], y=df_celltype['robustness_means'], yaxis='y2', offsetgroup=2)
#     #     ],
#     #     layout={
#     #         'yaxis': {'title': 'percentage'},
#     #         'yaxis2': {'title': 'robustness_means', 'overlaying': 'y', 'side': 'right'}
#     #     }
#     # )
#     #
#     # # Change the bar mode
#     # fig.update_layout(barmode='group')
#     # show chart
#     # fig.show()
#
#     subplot_titles = []
#     for type in set(df['id_type']):
#         for n_clusters in set(df['n_clusters']):
#             subplot_titles.append(f"{type}, n_clusters = {n_clusters}")
#     specs = [[{"secondary_y": True}] * 4] * 2
#
#     fig = make_subplots(rows=2, cols=4, subplot_titles=subplot_titles, specs=specs,
#                         vertical_spacing=0.3)
#     row = col = 1
#
#     for n_clusters in list(set(df['n_clusters'])) * 2:
#
#         if row == 1:
#             df_for_plot = df_organ[df_organ['n_clusters'] == n_clusters]
#         else:
#             df_for_plot = df_celltype[df_celltype['n_clusters'] == n_clusters]
#
#         bar1 = go.Bar(x=df_for_plot['new_id'], y=df_for_plot['percentage'], name="percentage", legendgroup="percentage",
#                marker=dict(color="blue"), yaxis='y',  offsetgroup=1)
#         ## px.bar(df_for_plot, x='new_id', y='percentage')
#         fig.add_trace(bar1, row=row, col=col, secondary_y=False)
#
#         bar2 = go.Bar(x=df_for_plot['new_id'], y=df_for_plot['robustness_means'], name="robustness_means",
#                marker=dict(color="paleturquoise"), legendgroup="robustness_means", yaxis='y2', offsetgroup=2)
#             # px.bar(df_for_plot, x='new_id', y='robustness_means')
#
#         fig.add_trace(bar2, row=row, col=col, secondary_y=True)
#
#         if col != 1 or row != 1:
#             fig.update_traces(showlegend=False, row=row, col=col)
#         if col == 4:
#             col = 1
#             row += 1
#         else:
#             col += 1
#
#     fig.update_layout(barmode='group')
#     # Set x-axis title
#     fig.update_xaxes(title_text='New Id')
#
#     # Set y-axis titles
#     fig.update_yaxes(title_text='Percentage', secondary_y=False, range=[0, 12])
#     fig.update_yaxes(title_text='robustness_means', secondary_y=True, range=[0, 1])
#     fig.update_layout(yaxis={'title': 'Percentage'},
#                       yaxis2={'title': 'robustness_means', 'overlaying': 'y', 'side': 'right'})
#
#     # Set plot title
#     fig.update_layout(title_text='Scores for diff n_clusters', legend_title_text='Scores')
#     fig.show()

def count_new_labels(path="../output/all_new_labels.csv"):
    df_all_labels = pd.read_csv(path)
    df = None
    for n_clusters, group in df_all_labels.groupby('n_clusters'):
        group_new_id_counts = group['new_id'].value_counts(normalize=True)
        robustness_means = []
        for id_ in group_new_id_counts.index:
            new_id_value_mean = group[group['new_id'] == id_]['value'].mean()
            robustness_means.append(new_id_value_mean)

        percentages = np.array(group_new_id_counts.values) * 100
        df_group_new_id_counts = pd.DataFrame({'new_id': group_new_id_counts.index,
                                               'percentage': percentages,
                                               'robustness_means': robustness_means})
        df_group_new_id_counts['n_clusters'] = n_clusters

        if df is not None:
            df = pd.concat([df, df_group_new_id_counts], ignore_index=True)
        else:
            df = df_group_new_id_counts

    df = df[['n_clusters', 'new_id', 'robustness_means', 'percentage']]
    # add columns for figure
    celltypes = ['B1', 'CYCLING_B', 'IMMATURE_B', 'LARGE_PRE_B', 'LATE_PRO_B', 'MATURE_B', 'PLASMA_B', 'PRE_PRO_B',
                 'PRO_B', 'SMALL_PRE_B']
    organs = ['BM', 'GU', 'KI', 'LI', 'MLN', 'SK', 'SP', 'TH', 'YS']
    df['id_type'] = df['new_id'].apply(lambda x: 'celltype' if x in celltypes else 'organ')

    df = df[df['n_clusters'] > 8]
    df = df.sort_values(by=['n_clusters', 'id_type', "percentage", "robustness_means"], ascending=False)
    df_celltype = df[df['id_type'] == 'celltype']
    df_organ = df[df['id_type'] == 'organ']

    subplot_titles = []
    for type in set(df['id_type']):
        for n_clusters in set(df['n_clusters']):
            subplot_titles.append(f"{type}, n_clusters = {n_clusters}")
    specs = [[{"secondary_y": True}] * 4] * 2

    fig = make_subplots(rows=2, cols=4, subplot_titles=subplot_titles, specs=specs,
                        vertical_spacing=0.3)
    row = col = 1
    for n_clusters in list(set(df['n_clusters'])) * 2:

        if row == 1:
            df_for_plot = df_organ[df_organ['n_clusters'] == n_clusters]
        else:
            df_for_plot = df_celltype[df_celltype['n_clusters'] == n_clusters]

        bar1 = go.Bar(x=df_for_plot['new_id'], y=df_for_plot['percentage'], name="percentage", legendgroup="percentage",
               marker=dict(color="blue"), yaxis='y',  offsetgroup=1)
        fig.add_trace(bar1, row=row, col=col, secondary_y=False)

        bar2 = go.Bar(x=df_for_plot['new_id'], y=df_for_plot['robustness_means'], name="robustness_means",
               marker=dict(color="paleturquoise"), legendgroup="robustness_means", yaxis='y2', offsetgroup=2)
        fig.add_trace(bar2, row=row, col=col, secondary_y=True)

        if col != 1 or row != 1:
            fig.update_traces(showlegend=False, row=row, col=col)
        if col == 4:
            col = 1
            row += 1
        else:
            col += 1

    fig.update_layout(barmode='group')
    # Set x-axis title
    fig.update_xaxes(title_text='New Id')
    # Set y-axis titles
    fig.update_yaxes(title_text='Percentage', secondary_y=False, range=[0, 12])
    fig.update_yaxes(title_text='robustness_means', secondary_y=True, range=[0, 1])
    fig.update_layout(yaxis={'title': 'Percentage'},
                      yaxis2={'title': 'robustness_means', 'overlaying': 'y', 'side': 'right'})
    # Set plot title
    fig.update_layout(title_text='Scores for diff n_clusters', legend_title_text='Scores')
    fig.show()


def create_latent_space_umap(df, transf_embeddings_attributes, id_):
    # calc pca for UMAP
    pca = sc.tl.pca(transf_embeddings_attributes)
    mapper_latent = umap.UMAP().fit_transform(transf_embeddings_attributes)
    df_for_umap = pd.DataFrame(mapper_latent, columns=["umap1", "umap2"])
    df['umap1'] = df_for_umap['umap1']
    df['umap2'] = df_for_umap['umap2']

    df["celltype"] = switch_to_celltype_fullname(df["celltype"])
    df["organ"] = switch_to_organ_fullname(df["organ"])

    for i in range(pca.shape[1]):
        df[f"pc{i + 1}"] = pca[:, i]

    # create needed df
    # attributes_map = get_attributes_map(model)
    # dfs = {}
    # columns_latent = [f"latent{i}" for i in range(1, n_latent_attribute_categorical+1)]
    # for attribute_ in transf_embeddings_attributes_ind:
    #     dfs[attribute_] = pd.DataFrame(
    #         transf_embeddings_attributes_ind[attribute_],
    #         columns=columns_latent)
    #     dfs[attribute_][attribute_] = list(attributes_map[attribute_].keys())
    #     dfs[attribute_][attribute_ + '_key'] = list(attributes_map[attribute_].values())
    #     df[attribute_ + "_key"] = df[attribute_].map(attributes_map[attribute_])

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))

    sns.scatterplot(
        data=df,
        x="umap1",
        y="umap2",
        hue="celltype",
        style="organ",
        ax=axs,
        alpha=.8,
        s=60,
        palette="deep"
    )

    axs.set_title("cell type")
    axs.set(xticklabels=[], yticklabels=[])
    axs.set_xlabel("UMAP1")
    axs.set_ylabel("UMAP2")
    axs.grid(False)
    axs.legend(loc="upper left", bbox_to_anchor=(1, 1), ncols=2)

    plt.tight_layout()
    plt.savefig(settings.FIG_DIR + f"cell_type{id_}.png", format="png", dpi=300)
    plt.show()


if __name__ == "__main__":
    settings.init()
    plot_robustness()
    # df_celltype_path = "../output/celltype_kmeans_models_scores.csv"
    # df_organ_path = "../output/organ_kmeans_models_scores.csv"
    # robustness(df_celltype_path, df_organ_path)

    # create_csv_new_labels()
    # count_new_labels()
    print('end')
