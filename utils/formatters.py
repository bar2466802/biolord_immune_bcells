################################################
# FILE: formatters.py
# WRITERS: Bar Melinarskiy
# DESCRIPTION: Formatters functions
################################################

import itertools
import numpy as np
import pandas as pd
# import biolord



def switch_to_celltype_fullname(col):
    col = col.replace(
        {
            "B1": "B1",
            "CYCLING_B": "Cycling B",
            "IMMATURE_B": "Immature B",
            "LARGE_PRE_B": "Large pre B",
            "LATE_PRO_B": "Late pro B",
            "MATURE_B": "Mature B",
            "PLASMA_B": "Palsma B",
            "PRE_PRO_B": "Pre pro B",
            "PRO_B": "Pro B",
            "SMALL_PRE_B": "Small pre B",
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


def get_attributes_map(model):
    return {
        "celltype": model.categorical_attributes_map["celltype"],
        "organ": model.categorical_attributes_map["organ"]
    }


def get_transf_embeddings_attributes(model):
    attributes_map = get_attributes_map(model)
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
    df = pd.DataFrame(df)
    for attribute_ in transf_embeddings_attributes_ind:
        df[attribute_ + "_key"] = df[attribute_].map(attributes_map[attribute_])

    df_attributes_map = {
        "attribute": [],
        "key": [],
        "value": []
    }
    for key in attributes_map:
        for k, v in attributes_map[key].items():
            df_attributes_map['attribute'].append(key)
            df_attributes_map['key'].append(k)
            df_attributes_map['value'].append(v)
    df_attributes_map = pd.DataFrame(df_attributes_map)
    print(f"df_attributes_map = {df_attributes_map}")
    # attributes_map_rev = pd.DataFrame(attributes_map_rev)
    return transf_embeddings_attributes, df, df_attributes_map, transf_embeddings_attributes_ind
