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
    model_groups = data.groupby(["Model"], sort=False, as_index=False)
    mean = model_groups.mean()
    standard_deviation = model_groups.std().add_suffix(" (STD)")
    model = mean["model"]
    mean = mean.drop(columns=["model"])
    standard_deviation = std.drop(columns=["model"])
    statistics = pd.concat([mean, standard_deviation], axis=1)[list(interleave([mean, standard_deviation]))]
    statistics.insert(0, 'model', model)
    return statistics


def postprocess():
    complete_data = pd.read_csv("complete_evaluation.csv")
    model_groups = dict(tuple(complete_data.groupby(["dataset", "predicate_source", "rule_type"])))
    print(model_groups)


if __name__ == '__main__':
    postprocess()