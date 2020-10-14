from datetime import datetime

import pandas as pd


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


def compute_stats(data):
    mean = data.groupby(["Model"], sort=False).mean().drop(columns=["split"]).add_prefix("Average ").reset_index()
    standard_deviation = data.groupby(["Model"], sort=False).std().drop(columns=["split"]).add_suffix(" (STD)").reset_index()
    model = mean["Model"]
    mean = mean.drop(columns=["Model"])
    standard_deviation = standard_deviation.drop(columns=["Model"])
    statistics = pd.concat([mean, standard_deviation], axis=1)[list(interleave([mean, standard_deviation]))]
    statistics.insert(0, 'Model', model)
    return statistics


def postprocess():
    complete_data = pd.read_csv("complete_evaluation.csv")
    model_groups = dict(tuple(complete_data.groupby(["dataset", "predicate_source", "rule_type"])))
    for model_group in model_groups:
        data = model_groups[model_group]
        num_splits = len(data["split"].unique())
        statistics = compute_stats(data)
        print(statistics.columns)


if __name__ == '__main__':
    postprocess()