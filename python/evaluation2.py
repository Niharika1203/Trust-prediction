"""Rewrite of the previous evaluation.py"""


MODELS = [
    "balance5",
    "balance5_recip",
    "balance_extended",
    "balance_extended_recip",
    "status" ,
    "status_inv" ,
    "personality",
    "cyclic_balanced",
    "cyclic_bal_unbal",
    "similarity",
    "triad-similarity",
    "triad-personality",
    "personality-similarity",
    "triad-pers-sim"
]


DATASETS = [
    "filmtrust",
    "epinions"
]

NUM_SPLITS = 8  # TODO: This may have to change if we use other papers' datasets


def main():
    evaluation = {
        "Dataset": [],
        "Model": [],
        "Squared Rules": [],
        "Split": [],
        "Predicate Source": [],  # Coupld be StarAI, SBP, ICML or JMLR
        "MAE": [],
        "MSE": [],
        "AU-PRC Positive Class": [],
        "AU-PRC Negative Class": [],
        "Spearman Correlation": [],
        "Kendall Coefficient": [],
        "AUC-ROC": []
    }
    for dataset in DATASETS:
        