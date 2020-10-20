from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn import metrics


DATASETS = [
    ("FilmTrust", "SBP-2013"),
    ("Epinions", "JMLR-2017")
]

NUM_SPLITS = 8  # TODO: This may have to change if we use other papers' datasets

RULE_TYPES = [
    "Linear",
    "Quadratic"
]

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

# FilmTrust_paper = {
# "balance5" : {"MAE" : 0.30 , "Spearman Correlation": 0.09, "Kendall Correlation" : 0.07 },
# "balance_extended" : {"MAE" :0.21 , "Spearman Correlation": 0.18, "Kendall Correlation" : 0.14 },
# "balance_extended_recip" : {"MAE" : 0.21 , "Spearman Correlation": 0.19, "Kendall Correlation" : 0.14 },
# "status" : {"MAE" : 0.22 , "Spearman Correlation": 0.14, "Kendall Correlation" : 0.11 },
# "status_inv" : {"MAE" : 0.22 , "Spearman Correlation": 0.09 , "Kendall Correlation" : 0.07 },
# "personality" : {"MAE" :0.26, "Spearman Correlation":0.23 , "Kendall Correlation" :0.17 },
# "similarity" : {"MAE" : 0.22 , "Spearman Correlation": 0.15, "Kendall Correlation" : 0.11 },
# "triad-similarity" : {"MAE" : 0.21 , "Spearman Correlation": 0.17, "Kendall Correlation" : 0.12 },
# "triad-personality" : {"MAE" : 0.25, "Spearman Correlation": 0.24, "Kendall Correlation" : 0.18 },
# "personality-similarity" : {"MAE" : 0.22 , "Spearman Correlation": 0.24, "Kendall Correlation" : 0.18 },
# "triad-pers-sim" : {"MAE" : 0.22 , "Spearman Correlation": 0.26, "Kendall Correlation" : 0.19 } }
#
# Epinions_paper = {
# "balance_extended" : {"AU-PRC Negative Class" :0.32 } ,
# "balance_extended_recip" : {"AU-PRC Negative Class" :0.34 },
# "status" : {"AU-PRC Negative Class" :0.30 } ,
# "status_inv" : {"AU-PRC Negative Class" :0.28 } }

def main():
    # Capitalize fields that will be kept in the final document as columns
    eval_dict = {
        "dataset": [],
        "predicate_source": [],
        "Model": [],
        "rule_type": [],
        "split": [],
        "MAE": [],
        # "MAE in Paper" : [],
        "Spearman Correlation": [],
        # "Spearman Correlation in Paper" : [],
        "Kendall Correlation": [],
        # "Kendall Correlation in Paper" : [],
        "AUC-ROC": [],
        # "AUC-ROC in Paper" : [],
        "AU-PRC Positive Class": [],
        # "AU-PRC Positive Class in Paper": [],
        "AU-PRC Negative Class": [],
        # "AU-PRC Negative Class in Paper": [],
    }
    for dataset, predicate_source in DATASETS:
        # TODO: Change folder names output by psl-models.py to match dataset names
        # TODO: Add Epinions predicates from SBP, make that part of folder name
        data_folder_name = 'film-trust' if (dataset == 'FilmTrust') else 'trust-prediction'
        # data_dictionary = FilmTrust_paper if (dataset == 'FilmTrust') else Epinions_paper
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
                    # if model in data_dictionary :
                    #     if "MAE" in data_dictionary[model] :
                    #         mae_paper = data_dictionary[model]["MAE"]
                    #     else : mae_paper = None
                    #     if "Spearman Correlation" in data_dictionary[model] : rho_paper = data_dictionary[model]["Spearman Correlation"]
                    #     else : rho_paper = None
                    #     if "Kendall Correlation" in data_dictionary[model] : tau_paper = data_dictionary[model]["Kendall Correlation"]
                    #     else: tau_paper  = None
                    #     if "AUC-ROC" in data_dictionary[model] : auroc_paper = data_dictionary[model]["AUC-ROC"]
                    #     else :auroc_paper  = None
                    #     if "AU-PRC Positive Class" in data_dictionary[model] : aupr_pos_paper = data_dictionary[model]["AU-PRC Positive Class"]
                    #     else : aupr_pos_paper  = None
                    #     if "AU-PRC Negative Class" in data_dictionary[model] : aupr_neg_paper = data_dictionary[model]["AU-PRC Negative Class"]
                    #     else : aupr_neg_paper = None

                    # eval_dict["MAE in Paper"].append(mae_paper)
                    # print(mae_paper)
                    eval_dict["Spearman Correlation"].append(stats.spearmanr(truth_array, prediction_array)[0])
                    # eval_dict["Spearman Correlation in Paper"].append(rho_paper)
                    eval_dict["Kendall Correlation"].append(stats.kendalltau(truth_array, prediction_array)[0])
                    # eval_dict["Kendall Correlation in Paper"].append(tau_paper)
                    truth_array_threshold = truth_array > 0.5
                    eval_dict["AUC-ROC"].append(metrics.roc_auc_score(truth_array_threshold, prediction_array))
                    eval_dict["AU-PRC Positive Class"].append(metrics.average_precision_score(truth_array_threshold, prediction_array))
                    eval_dict["AU-PRC Negative Class"].append(metrics.average_precision_score(1-truth_array_threshold, 1-prediction_array))
                    # eval_dict["AUC-ROC in Paper"].append(auroc_paper)
                    # eval_dict["AU-PRC Positive Class in Paper"].append(aupr_pos_paper)
                    # eval_dict["AU-PRC Negative Class in Paper"].append(aupr_neg_paper)

    complete_evaluation = pd.DataFrame(eval_dict)
    group_eval(complete_evaluation)


def group_eval(complete_data):
    model_groups = dict(tuple(complete_data.groupby(["dataset", "predicate_source", "rule_type"], sort=False)))
    current_line = 0
    title_spacing = 1
    dataset_spacing = 2
    title_lines = []
    dataset_widths = []

    timestamp = datetime.now().strftime("%d-%b-%y %-I.%M %p")
    sheet_name = f"Evaluation ({timestamp})"

    eval_dir = RESULTS_DIR / "evaluation"
    eval_dir.mkdir(exist_ok=True, parents=True)
    fname = eval_dir / "evaluation.xlsx"
    with pd.ExcelWriter(fname) as writer:
        for (dataset, predicate_source, rule_type), data in model_groups.items():
            print(f"Aggregating:", dataset, predicate_source, rule_type)
            num_splits = len(data["split"].unique())

            statistics = compute_stats(data)
            # statistics = statistics[statistics.columns.drop(list(statistics.filter(regex='Paper.*STD')))]

            title_line = current_line
            current_line += title_spacing
            width = len(statistics.columns)
            height = len(statistics) + 1

            title_lines.append(title_line)
            dataset_widths.append(width)
            statistics.to_excel(writer, sheet_name=sheet_name, startrow=current_line, index=False, float_format="%.4f", na_rep='N/A')

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            worksheet.merge_range(
                title_line, 0, title_line, width-1,
                f"{dataset}: Average of {num_splits} splits from {predicate_source} (with {rule_type} rules)"
            )
            current_line += height + dataset_spacing
        big_font = workbook.add_format({'font_size': 15})
        for line in title_lines:
            worksheet.set_row(line, None, big_font)
        for col in range(max(dataset_widths)):
            worksheet.set_column(col, col, 25)
    print(f"Wrote:", fname)


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
