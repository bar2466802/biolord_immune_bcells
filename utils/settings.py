import scanpy as sc
import os
from utils.train_model import split_adata_into_train_test
import random

def init():
    global DATA_DIR, SAVE_DIR, LOGS_DIR, FIG_DIR, DIR
    global adata
    DATA_DIR = "../data/"
    SAVE_DIR = "../output/"
    LOGS_DIR = f"../logs/"
    FIG_DIR = f"../figures/"

def init_adata(new_dir_name=1):
    global DATA_DIR, SAVE_DIR, LOGS_DIR, FIG_DIR, DIR
    global adata
    DATA_DIR = f"../data/{new_dir_name}/"
    SAVE_DIR = f"../output/{new_dir_name}/"
    LOGS_DIR = f"../logs/{new_dir_name}/"
    FIG_DIR = f"../figures/{new_dir_name}/"
    DIR = new_dir_name
    adata = sc.read(DATA_DIR + "biolord_immune_bcells_bm.h5ad")
    # re-split the adata file
    split_adata_into_train_test()
    adata.write(DATA_DIR + f"/{new_dir_name}_biolord_immune_bcells_bm.h5ad")


def init_folders(new_dir_name=1):
    global DATA_DIR, SAVE_DIR, LOGS_DIR, FIG_DIR, DIR, LOGS_CSV
    DATA_DIR = f"../data/{new_dir_name}/"
    SAVE_DIR = f"../output/{new_dir_name}/"
    LOGS_DIR = f"../logs/{new_dir_name}/"
    FIG_DIR = f"../figures/{new_dir_name}/"
    LOGS_CSV = SAVE_DIR + "/trained_model_scores.csv"
    DIR = new_dir_name

    for dir_path in [SAVE_DIR, FIG_DIR, LOGS_DIR]:
        if not os.path.exists(dir_path + new_dir_name):
            os.makedirs(dir_path + new_dir_name)

    random.seed(42)



