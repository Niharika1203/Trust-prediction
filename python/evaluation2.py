"""Rewrite of the previous evaluation.py"""

from pathlib import Path
import pandas as pd
from sklearn import metrics
from scipy import stats


DATASETS = [
    ("FilmTrust", "SBP-2013"),
    ("Epinions", "JMLR-2017")
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
    group_eval(complete_evaluation)


def group_eval(complete_data):
    model_groups = dict(tuple(complete_data.groupby(["dataset", "predicate_source", "rule_type"])))
    for model_group in model_groups:
        data = model_groups[model_group]
        num_splits = len(data["split"].unique())
        statistics = compute_stats(data)
        eval_dir = RESULTS_DIR / "evaluation"
        eval_dir.mkdir(exist_ok=True, parents=True)
        fname = eval_dir / ("_".join(model_group) + '.txt')
        statistics.to_csv(fname, index=False)
        print("Saved: ", fname)


def compute_stats(data):
    mean = data.groupby(["Model"], sort=False).mean().drop(columns=["split"]).add_prefix("Average ").reset_index()
    standard_deviation = data.groupby(["Model"], sort=False).std().drop(columns=["split"]).add_suffix(" (STD)").reset_index()
    model = mean["Model"]
    mean = mean.drop(columns=["Model"])
    standard_deviation = standard_deviation.drop(columns=["Model"])
    statistics = pd.concat([mean, standard_deviation], axis=1)[list(interleave([mean, standard_deviation]))]
    statistics.insert(0, 'Model', model)
    return statistics


def interleave(seqs):
    """ Interleave a sequence of sequences

    >>> list(interleave([[1, 2], [3, 4]]))
    [1, 3, 2, 4]

    >>> ''.join(interleave(('ABC', 'XY')))
    'AXBYC'

    Both the individual sequences and the sequence of sequences may be infinite

    Returns a lazy iterator
    """
    import functools
    import itertools
    import operator
    # Taken from toolz
    iters = itertools.cycle(map(iter, seqs))
    while True:
        try:
            for itr in iters:
                yield next(itr)
            return
        except StopIteration:
            predicate = functools.partial(operator.is_not, itr)
            iters = itertools.cycle(itertools.takewhile(predicate, iters))


if __name__ == '__main__':
    main()