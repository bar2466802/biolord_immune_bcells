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
