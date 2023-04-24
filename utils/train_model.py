################################################
# FILE: train_model.py
# WRITERS: Bar Melinarskiy
# DESCRIPTION: train biolord model with the given settings - create logs for metrics and k means model
################################################

import biolord
import scanpy as sc
import anndata
import numpy as np
import pandas as pd
import argparse
from os.path import exists
import torch
import umap.plot
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import biolord
# import wandb
import os
import sys
import settings

sys.path.append("/cs/usr/bar246802/bar246802/SandBox2023/biolord_immune_bcells/utils")  # add utils
sys.path.append("/cs/usr/bar246802/bar246802/SandBox2023/biolord_immune_bcells/")  # add utils
sys.path.append("/cs/usr/bar246802/bar246802/SandBox2023/biolord")  # set path)
from cluster_analysis import *
from formatters import *
from analysis.analyze_logs import *

print(f"PyTorch version: {torch.__version__}")
# Set the device
device = "gpu" if torch.backends.cuda.is_built() else "cpu"
print(f"Using device: {device}")

from tqdm import tqdm

tqdm(disable=True, total=0)  # initialise internal lock

import mplscience

mplscience.set_style()
plt.rcParams['legend.scatterpoints'] = 1


def cluster_evaluate(model, id_, attributes=['celltype', 'organ']):
    transf_embeddings_attributes, df, df_attributes_map, transf_embeddings_attributes_ind = get_transf_embeddings_attributes(model)
    attributes_ground_truth_labels = {'attributes': [], 'true_labels': []}
    for attribute in attributes:
        ground_truth_labels = np.array(df[attribute + '_key'])
        attributes_ground_truth_labels['attributes'].append(attribute)
        attributes_ground_truth_labels['true_labels'].append(ground_truth_labels)
        ground_truth_unique_labels = list(set(ground_truth_labels))
        print(f'For attribute {attribute} the # of unique true labels is: {len(ground_truth_unique_labels)}')

    path = settings.SAVE_DIR + "kmeans_models_scores.csv"
    n_clusters_range = np.arange(9, 13).astype(int)
    scores = get_kmeans_score(transf_embeddings_attributes, attributes_ground_truth_labels,
                              n_clusters_range=n_clusters_range, id_=id_, save_path=path)
    print(scores)
    attributes_map_name = settings.SAVE_DIR + f"attributes_map_{id_}.csv"
    df_attributes_map.to_csv(attributes_map_name)
    attributes_ground_truth_labels = pd.DataFrame(attributes_ground_truth_labels)
    attributes_true_labels_name = settings.SAVE_DIR + f"attributes_true_labels_{id_}.csv"
    attributes_ground_truth_labels.to_csv(attributes_true_labels_name)
    create_latent_space_umap(df, transf_embeddings_attributes, id_)
    return scores


def train_model(module_params, trainer_params):
    # before each train we wish to re-split the data to make sure we are not biased to a certain split
    model = biolord.Biolord(
        adata=settings.adata,
        n_latent=32,
        model_name="immune_bcells",
        module_params=module_params,
        train_classifiers=False,
        split_key="split",
    )

    model.train(max_epochs=1000,
                use_gpu=True,
                batch_size=512,
                plan_kwargs=trainer_params,
                early_stopping=True,
                early_stopping_patience=20,
                check_val_every_n_epoch=10,
                enable_checkpointing=False,
                num_workers=1)
    return model


def train_dataset():
    """
    Read arguments if this script is called from a terminal.
    """
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--n_latent_attribute_categorical", type=int)
    parser.add_argument("--reconstruction_penalty", type=float)
    parser.add_argument("--unknown_attribute_penalty", type=float)
    parser.add_argument("--unknown_attribute_noise_param", type=float)
    parser.add_argument("--id_", type=int)
    parser.add_argument("--folder", type=int)
    args = parser.parse_args()
    print('*****************************************************************************')
    print(f'args = {args}')
    print(
        f'n_latent_attribute_categorical = {args.n_latent_attribute_categorical}, \nreconstruction_penalty = {args.reconstruction_penalty}, \nunknown_attribute_penalty = {args.unknown_attribute_penalty}, \nunknown_attribute_noise_param = {args.unknown_attribute_noise_param}, \nid_= {args.id_}, \nfolder = {args.folder}'
    )
    settings.init_folders(args.folder)
    settings.adata = sc.read(settings.DATA_DIR + f"{args.folder}_biolord_immune_bcells_bm.h5ad")

    biolord.Biolord.setup_anndata(
        settings.adata,
        categorical_attributes_keys=["celltype", "organ", "age"],
        retrieval_attribute_key="sex",
    )

    module_params = {
        "autoencoder_width": 128,
        "autoencoder_depth": 2,
        "attribute_nn_width": 256,
        "attribute_nn_depth": 2,
        "n_latent_attribute_categorical": args.n_latent_attribute_categorical,
        "loss_ae": "gauss",
        "loss_ordered_attribute": "gauss",
        "reconstruction_penalty": args.reconstruction_penalty,
        "unknown_attribute_penalty": args.unknown_attribute_penalty,
        "unknown_attribute_noise_param": args.unknown_attribute_noise_param,
        "attribute_dropout_rate": 0.1,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "seed": 42,
    }

    trainer_params = {
        "n_epochs_warmup": 0,
        "autoencoder_lr": 1e-4,
        "autoencoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2,
        "attribute_nn_wd": 4e-8,
        "step_size_lr": 45,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }
    # wandb.init(project="biolord_bcells_train", entity="biolord", config=trainer_params)
    # wandb.log({'n_latent_attribute_categorical': args.n_latent_attribute_categorical})
    # wandb.log({'reconstruction_penalty': args.reconstruction_penalty})
    # wandb.log({'unknown_attribute_penalty': args.unknown_attribute_penalty})
    # wandb.log({'unknown_attribute_noise_param': args.unknown_attribute_noise_param})
    # wandb.log({'id_': args.id_})

    model = train_model(module_params, trainer_params)
    model.save(settings.SAVE_DIR + "trained_model_" + str(args.id_), overwrite=True)
    settings.init_folders(args.folder)
    scores = cluster_evaluate(model, args.id_)
    scores['n_latent_attribute_categorical'] = args.n_latent_attribute_categorical
    scores['reconstruction_penalty'] = args.reconstruction_penalty
    scores['unknown_attribute_penalty'] = args.unknown_attribute_penalty
    scores['unknown_attribute_noise_param'] = args.unknown_attribute_noise_param
    # scores['id_'] = args.id_
    scores = pd.DataFrame(scores)
    # for index, score_row in scores.iterrows():
    #     wandb.log({
    #         f"attribute_{score_row['attribute']}_score_{score_row['score_name']}": score_row['score'],
    #     })

    if not exists(settings.LOGS_CSV):
        scores.to_csv(settings.LOGS_CSV)
    else:
        scores.to_csv(settings.LOGS_CSV, mode='a', header=False)


if __name__ == "__main__":
    print(sys.argv)
    train_dataset()
