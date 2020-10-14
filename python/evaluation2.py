"""Rewrite of the previous evaluation.py"""

from pathlib import Path
import pandas as pd
from sklearn import metrics
from scipy import stats


DATASETS = [
    ("FilmTrust", "SBP 2013"),
    ("Epinions", "JMLR 2017")
]
NUM_SPLITS = 8  # TODO: This may have to change if we use other papers' datasets
RULE_TYPES = ["Linear", "Quadratic"]
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
    "triad-personality",
    "similarity",
    "triad-similarity",
    "personality-similarity",
    "triad-pers-sim"
]

FILMTRUST_SPECIFIC = {"similarity", "triad-similarity", "personality-similarity", "triad-pers-sim"}
RESULTS_DIR = Path(__file__).parent.absolute()
DATA_DIR = RESULTS_DIR.parent / "data"



def main():
    # Capitalize fields that will be kept in the final document as columns
    eval_dict = {
        "dataset": [],
        "predicate_source": [],
        "Model": [],
        "rule_type": [],
        "split": [],
        "MAE": [],
        # "MSE": [],
        "Spearman Correlation": [],
        "Kendall Correlation": [],
        "AUC-ROC": [],
        "AU-PRC Positive Class": [],
        "AU-PRC Negative Class": [],
    }
    for dataset, predicate_source in DATASETS:
        # TODO: Change folder names output by psl-models.py to match dataset names
        # TODO: Add Epinions predicates from SBP, make that part of folder name
        data_folder_name = 'film-trust' if (dataset == 'FilmTrust') else 'trust-prediction'
        for split in range(NUM_SPLITS):
            truth_file = DATA_DIR / data_folder_name / str(split) / "eval" / "trusts_truth.txt"
            truth = pd.read_csv(truth_file, names=["u1", "u2", "truth"], sep='\t')
            truth_array = truth["truth"].to_numpy()
            for model in MODELS:
                if dataset == "Epinions" and model in FILMTRUST_SPECIFIC:
                    continue
                for rule_type in RULE_TYPES:
                    print("Processing:", dataset, split, rule_type, model)
                    rule_type_folder = "squared_True" if (rule_type == "Quadratic") else "squared_False"  # TODO: Fix these to reflect rule type names
                    predicted_file = RESULTS_DIR / data_folder_name / rule_type_folder / model / str(split) / "inferred-predicates" / "TRUSTS.txt"
                    predicted = pd.read_csv(predicted_file, names=["u1", "u2", "predicted"], sep='\t')
                    # Sort predictions by order of truth and transform both to arrays for ease of use
                    data = truth.merge(predicted, on=["u1", "u2"])
                    prediction_array = data["predicted"].to_numpy()
                    eval_dict["dataset"].append(dataset)
                    eval_dict["predicate_source"].append(predicate_source)
                    eval_dict["Model"].append(model)
                    eval_dict["rule_type"].append(rule_type)
                    eval_dict["split"].append(split)
                    eval_dict["MAE"].append(metrics.mean_absolute_error(truth_array, prediction_array))
                    eval_dict["Spearman Correlation"].append(stats.spearmanr(truth_array, prediction_array)[0])
                    eval_dict["Kendall Correlation"].append(stats.kendalltau(truth_array, prediction_array)[0])
                    # TODO: For now, don't try to threshold FilmTrust to get these scores
                    if dataset == 'Epinions':
                        eval_dict["AUC-ROC"].append(metrics.roc_auc_score(truth_array, prediction_array))
                        eval_dict["AU-PRC Positive Class"].append(metrics.average_precision_score(truth_array, prediction_array))
                        eval_dict["AU-PRC Negative Class"].append(metrics.average_precision_score(1-truth_array, 1-prediction_array))
                    else:
                        eval_dict["AUC-ROC"].append(None)
                        eval_dict["AU-PRC Positive Class"].append(None)
                        eval_dict["AU-PRC Negative Class"].append(None)
    complete_evaluation = pd.DataFrame(eval_dict).sort_values(["dataset", "predicate_source", "Model", "rule_type", "split"]).reset_index(drop=True)
    complete_evaluation.to_csv("complete_evaluation.csv", index=False)
    print(complete_evaluation)


if __name__ == '__main__':
    main()