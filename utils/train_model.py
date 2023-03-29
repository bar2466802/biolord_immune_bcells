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
import wandb
import os
import sys
import random

sys.path.append("/cs/usr/bar246802/bar246802/SandBox2023/biolord_immune_bcells/utils") # add utils
sys.path.append("/cs/usr/bar246802/bar246802/SandBox2023/biolord") # set path)
from cluster_analysis import *
from formatters import *

print(f"PyTorch version: {torch.__version__}")
# Set the device
device = "gpu" if torch.backends.cuda.is_built() else "cpu"
print(f"Using device: {device}")

from tqdm import tqdm
tqdm(disable=True, total=0)  # initialise internal lock

import mplscience
mplscience.set_style()
plt.rcParams['legend.scatterpoints'] = 1

DATA_DIR = "../data/"
SAVE_DIR = "../output/"
FIG_DIR = "../figures/"
LOGS_CSV = SAVE_DIR + "trained_model_scores.csv"

adata = sc.read(DATA_DIR + "biolord_immune_bcells_bm.h5ad")
random.seed(42)

def cluster_evaluate(model, id_, attributes = ['celltype', 'organ']):
    transf_embeddings_attributes, df = get_transf_embeddings_attributes(model)
    all_scores = None
    for attribute in attributes:
        ground_truth_labels = np.array(df[attribute + '_key'])
        ground_truth_unique_labels = list(set(ground_truth_labels))
        print(f'For attribute {attribute} the # of unique true labels is: {len(ground_truth_unique_labels)}')
        path = SAVE_DIR + attribute + "_"
        n_clusters_range = np.arange(2, 16).astype(int)
        scores = get_kmeans_score(transf_embeddings_attributes, ground_truth_labels, n_clusters_range=n_clusters_range, id_=id_, save_path=path)
        scores['attribute'] = attribute
        if all_scores is not None:
            all_scores = pd.concat([all_scores, scores], ignore_index=True)
        else:
            all_scores = scores
    cols = ['attribute', 'score_name', 'score', 'n_clusters']
    all_scores = all_scores[cols]
    print(all_scores)
    return all_scores


def split_adata_into_train_test():
    from sklearn.model_selection import train_test_split
    adata.obs['split'] = 'nan'
    ood_samples = adata.obs.sample(frac = 0.0025, random_state=42).index
    adata.obs.loc[ood_samples, "split"] = 'ood'

    adata_idx = adata.obs_names[adata.obs["split"] != 'ood']
    adata_idx_train, adata_idx_test = train_test_split(adata_idx, test_size=0.1, random_state=42)
    adata.obs.loc[adata_idx_train, "split"] = 'train'
    adata.obs.loc[adata_idx_test, "split"] = 'test'
    a = adata.obs['split'].value_counts()
    print("Simaple value count of train, test, OOD:")
    print(a)
    print("\n")
    print("Train, test, OOD by percentage:")
    p = adata.obs['split'].value_counts(normalize=True) * 100
    print(p)


def train_model(module_params, trainer_params):
    # before each train we wish to re-split the data to make sure we are not biased to a certain split
    split_adata_into_train_test()
    model = biolord.Biolord(
        adata=adata,
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
    args = parser.parse_args()

    print(
        f'n_latent_attribute_categorical = {args.n_latent_attribute_categorical}, reconstruction_penalty = {args.reconstruction_penalty},unknown_attribute_penalty = {args.unknown_attribute_penalty}, unknown_attribute_noise_param = {args.unknown_attribute_noise_param}, id_= {args.id_}'
    )

    biolord.Biolord.setup_anndata(
        adata,
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
    wandb.init(project="biolord_bcells_train", entity="biolord", config=trainer_params)
    wandb.log({'n_latent_attribute_categorical': args.n_latent_attribute_categorical})
    wandb.log({'reconstruction_penalty': args.reconstruction_penalty})
    wandb.log({'unknown_attribute_penalty': args.unknown_attribute_penalty})
    wandb.log({'unknown_attribute_noise_param': args.unknown_attribute_noise_param})
    wandb.log({'id_': args.id_})

    model = train_model(module_params, trainer_params)
    scores = cluster_evaluate(model, args.id_)
    scores['n_latent_attribute_categorical'] = args.n_latent_attribute_categorical
    scores['reconstruction_penalty'] = args.reconstruction_penalty
    scores['unknown_attribute_penalty'] = args.unknown_attribute_penalty
    scores['unknown_attribute_noise_param'] = args.unknown_attribute_noise_param
    scores['id_'] = args.id_
    scores = pd.DataFrame(scores)
    for index, score_row in scores.iterrows():
        wandb.log({
            f"attribute_{score_row['attribute']}_score_{score_row['score_name']}":  score_row['score'],
        })

    if args.id_ == 1 or not exists(LOGS_CSV):
        scores.to_csv(LOGS_CSV)
    else:
        scores.to_csv(LOGS_CSV, mode='a', header=False)

if __name__ == "__main__":
    train_dataset()
