################################################
# FILE: formatters.py
# WRITERS: Bar Melinarskiy
# DESCRIPTION: Formatters functions
################################################

import itertools
import numpy as np
import pandas as pd
import biolord



def switch_to_celltype_fullname(col):
    col = col.replace(
        {
            "B1": "B1",
            "CYCLING_B": "cycling B",
            "IMMATURE_B": "immature B",
            "LARGE_PRE_B": "large pre B",
            "LATE_PRO_B": "late pro B",
            "MATURE_B": "mature B",
            "PLASMA_B": "palsma B",
            "PRE_PRO_B": "pre pro B",
            "PRO_B": "pro B",
            "SMALL_PRE_B": "small pre B",
        }
    )
    return col


def switch_to_organ_fullname(col):
    col = col.replace(
        {
            "BM": "Bone Marrow",
            "GU": "Gut",
            "KI": "Kidney",
            "LI": "Liver",
            "MLN": "Lymph Node",
            "SK": "Skin",
            "SP": "Spleen",
            "TH": "Thymus",
            "YS": "Yolk Sac",
        }
    )
    return col

def get_transf_embeddings_attributes(model):
    attributes_map = {
        "celltype": model.categorical_attributes_map["celltype"],
        "organ": model.categorical_attributes_map["organ"]
    }

    transf_embeddings_attributes = {
        attribute_:
            model.get_categorical_attribute_embeddings(attribute_key=attribute_)
        for attribute_ in model.categorical_attributes_map
    }

    keys = list(
        itertools.product(*[
            list(model.categorical_attributes_map[attribute_].keys())
            for attribute_ in model.categorical_attributes_map
        ]))

    transf_embeddings_attributes = [
        np.concatenate(([
            transf_embeddings_attributes[map_[0]][map_[1][key_[ci]], :]
            for ci, map_ in enumerate(model.categorical_attributes_map.items())
        ]), 0) for key_ in keys
    ]

    transf_embeddings_attributes_ind = {
        attribute_:
            model.get_categorical_attribute_embeddings(attribute_key=attribute_)
        for attribute_ in attributes_map
    }

    keys = list(
        itertools.product(*[
            list(model.categorical_attributes_map[attribute_].keys())
            for attribute_ in attributes_map
        ]))

    transf_embeddings_attributes = [
        np.concatenate(([
            transf_embeddings_attributes_ind[map_[0]][map_[1][key_[ci]], :]
            for ci, map_ in enumerate(attributes_map.items())
        ]), 0) for key_ in keys
    ]

    transf_embeddings_attributes = np.asarray(transf_embeddings_attributes)

    df = {}
    cols = {
        attribute_: [key_[ci] for key_ in keys]
        for ci, attribute_ in enumerate(attributes_map)
    }
    for col_, map_ in cols.items():
        df[col_] = map_
    for attribute_ in transf_embeddings_attributes_ind:
        df[attribute_ + "_key"] = df[attribute_].map(attributes_map[attribute_])

    return transf_embeddings_attributes, df
