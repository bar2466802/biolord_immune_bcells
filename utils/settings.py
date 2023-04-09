import scanpy as sc
import os
import random


def init():
    global DATA_DIR, SAVE_DIR, LOGS_DIR, FIG_DIR, DIR
    global adata
    DATA_DIR = "../data/"
    SAVE_DIR = "../output/"
    LOGS_DIR = f"../logs/"
    FIG_DIR = f"../figures/"


def init_adata(new_dir_name="1"):
    global DATA_DIR, SAVE_DIR, LOGS_DIR, FIG_DIR, DIR
    global adata
    DATA_DIR = "../data/"
    SAVE_DIR = f"../output/{new_dir_name}/"
    LOGS_DIR = f"../logs/{new_dir_name}/"
    FIG_DIR = f"../figures/{new_dir_name}/"
    DIR = new_dir_name
    adata = sc.read(DATA_DIR + "biolord_immune_bcells_bm.h5ad")
    # re-split the adata file
    split_adata_into_train_test()
    adata.write(DATA_DIR + f"{new_dir_name}_biolord_immune_bcells_bm.h5ad")


def init_folders(new_dir_name="1"):
    global DATA_DIR, SAVE_DIR, LOGS_DIR, FIG_DIR, DIR, LOGS_CSV
    print(f'init_folders.new_dir_name = {new_dir_name}')
    new_dir_name = str(new_dir_name)
    DATA_DIR = "../data/"
    SAVE_DIR = f"../output/{new_dir_name}/"
    LOGS_DIR = f"../logs/{new_dir_name}/"
    FIG_DIR = f"../figures/{new_dir_name}/"
    LOGS_CSV = SAVE_DIR + "/trained_model_scores.csv"
    DIR = new_dir_name

    random.seed(42)


def split_adata_into_train_test():
    from sklearn.model_selection import train_test_split
    adata.obs['split'] = 'nan'
    ood_samples = adata.obs.sample(frac=0.0025, random_state=42).index
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

