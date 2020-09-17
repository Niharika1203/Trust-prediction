#!/usr/bin/env python3

import os
from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from pathlib import Path

# MODEL_NAME = 'psl-models'
DATA_DIR = os.path.join('..', 'data/trust-prediction/')

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

SPLITS = [0]

ADDITIONAL_CLI_OPTIONS = [
     # '--postgres'
]

def main():
    # model = Model(MODEL_NAME)
    for data_split in SPLITS :
        # predicate_dir = DATA_DIR + str(split) + "/eval"
        # print(predicate_dir)
        models = [ "balance5" ,"balance5_recip", "balance_extended", "balance_extended_recip",
          "status" , "status_inv" , "personality", "cyclic_balanced" , "cyclic_bal_unbal" , ]

        for model_name in models :
            model = makeModel(model_name)
            if model_name == "balance5" :
                balance5rules(model)
            elif model_name == "balance5_recip" :
                balance5rules(model, recip = True)
            elif model_name == "balance_extended" :
                balance5rules(model)
                balanceExtended(model)
            elif model_name == "balance_extended_recip" :
                balance5rules(model, recip = True)
                balanceExtended(model)
            elif model_name == "cyclic_balanced" :
                cyclic_bal_rules(model)
            elif model_name == "cyclic_bal_unbal" :
                cyclic_bal_rules(model, unbalanced = True)
            elif model_name == "status" :
                status_rules(model)
            elif model_name == "status_inv" :
                status_rules(model, inv = True)
            elif model_name == "personality" :
                personality_rules(model)
            else :
                print("No such model defined.")


            print('Rules defined:')
            for rule in model.get_rules():
                print('   ' + str(rule))

            # Weight Learning
            model = learn(model, str(data_split) , model_name)

            print('Learned Rules:')
            for rule in model.get_rules():
                print('   ' + str(rule))

            # Inference
            results = infer(model, str(data_split) , model_name)
            write_results(results, model, model_name)

def makeModel(model_name, addPrior = True, square = True, sim = False):
    model = Model(model_name)
    Trusts = Predicate("Trusts", size=2, closed=False)
    Knows = Predicate("Knows", size=2, closed=True)
    Prior = Predicate("Prior", size=1, closed=True)
    model.add_predicate(Trusts)
    model.add_predicate(Knows)
    model.add_predicate(Prior)

    if model_name == "personality" :
        Trusting = Predicate("Trusting", size = 1, closed=False)
        TrustWorthy = Predicate("TrustWorthy", size=1, closed = False)
        model.add_predicate(Trusting)
        model.add_predicate(TrustWorthy)

    # if addPrior :

    return model

def add_learn_data(model,datasplit, model_name):
    _add_data('learn', model, datasplit, model_name)

def add_eval_data(model,datasplit, model_name):
    _add_data('eval', model,datasplit, model_name )

def _add_data(split, model, datasplit, model_name):
    split_data_dir = os.path.join(DATA_DIR, datasplit, split)
    for predicate in model.get_predicates().values():
        predicate.clear_data()

    path = os.path.join(split_data_dir, 'trusts_obs.txt')
    model.get_predicate('Trusts').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(split_data_dir, 'knows_obs.txt')
    model.get_predicate('Knows').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(split_data_dir, 'prior_obs.txt')
    model.get_predicate('Prior').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(split_data_dir, 'trusts_target.txt')
    model.get_predicate('Trusts').add_data_file(Partition.TARGETS, path)

    path = os.path.join(split_data_dir, 'trusts_truth.txt')
    model.get_predicate('Trusts').add_data_file(Partition.TRUTH, path)

    if model_name == "personality":
        path = os.path.join(split_data_dir, 'users.txt')
        model.get_predicate('Trusting').add_data_file(Partition.TARGETS, path)
        path = os.path.join(split_data_dir, 'usersTW.txt')
        model.get_predicate('TrustWorthy').add_data_file(Partition.TARGETS, path)

def learn(model, datasplit, model_name):
    add_learn_data(model, datasplit, model_name)
    model.learn(additional_cli_optons = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)
    return model

def write_results(results, model, model_name):
    out_dir = model_name + '/inferred-predicates'
    os.makedirs(out_dir, exist_ok = True)

    for predicate in model.get_predicates().values():
        if (predicate.closed()):
            continue

        out_path = os.path.join(out_dir, "%s.txt" % (predicate.name()))
        results[predicate].to_csv(out_path, sep = "\t", header = False, index = False)

def balance5rules(model, recip = False):

    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & !Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & Trusts(A, B) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & Trusts(B, A) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))
    # two-sided prior
    model.add_rule(Rule("1.0: Knows(A, B) & Prior('0') -> Trusts(A, B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> Prior('0') ^2"))

    if recip :
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, A) & Trusts(A, B) -> Trusts(B, A) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, A) & !Trusts(A, B) -> !Trusts(B, A) ^2"))

def balanceExtended(model):
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & !Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & Trusts(A, B) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & !Trusts(A, B) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & !Trusts(A, B) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & Trusts(B, A) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & !Trusts(B, A) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & !Trusts(B, A) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & Trusts(B, A) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & Trusts(B, A) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & !Trusts(B, A) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & !Trusts(B, A) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))

def cyclic_bal_rules(model, unbalanced = False) :

    model.add_rule(Rule("1.0: Knows(A, B) & Prior('0') -> Trusts(A, B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> Prior('0') ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, A) & Trusts(A, B) -> Trusts(B, A) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, A) & !Trusts(A, B) -> !Trusts(B, A) ^2"))

    if not unbalanced :
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(C, A) & Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(C, A) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(C, A) & !Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(C, A) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(A, C) & Knows(C, B) & Trusts(A,B) & Trusts(A,C) & (A != B) & (B != C) & (A != C) -> Trusts(C,B) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(A, C) & Knows(C, B) & Trusts(A,B) & !Trusts(A,C) & (A != B) & (B != C) & (A != C) -> !Trusts(C,B) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(A, C) & Knows(C, B) & !Trusts(A,B) & Trusts(A,C) & (A != B) & (B != C) & (A != C) -> !Trusts(C,B) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & knows(A, C) & knows(C, B) & !Trusts(A,B) & !Trusts(A,C) & (A != B) & (B != C) & (A != C) -> Trusts(C,B) ^2"))

    if unbalanced :
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(C, A) & !Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(C, A) ^2"))
        # model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(C, A) & !Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(C, A) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(A, C) & Knows(C, B) & Trusts(A,B) & Trusts(A,C) & (A != B) & (B != C) & (A != C) -> !Trusts(C,B) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(A, C) & Knows(C, B) & Trusts(A,B) & !Trusts(A,C) & (A != B) & (B != C) & (A != C) -> Trusts(C,B) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(A, C) & Knows(C, B) & !Trusts(A,B) & Trusts(A,C) & (A != B) & (B != C) & (A != C) ->  Trusts(C,B) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & knows(A, C) & knows(C, B) & !Trusts(A,B) & !Trusts(A,C) & (A != B) & (B != C) & (A != C) -> !Trusts(C,B) ^2"))


def status_rules(model, inv = False) :
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & !Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & Trusts(A, B) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))

    model.add_rule(Rule("1.0: Knows(A, B) & Knows(C, B) & Knows(A, C) & !Trusts(A, B) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & Trusts(B, A) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(B, C) & Knows(A, C) & !Trusts(B, A) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))

    model.add_rule(Rule("1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & Trusts(B, A) & Trusts(C, B) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(B, A) & Knows(C, B) & Knows(A, C) & !Trusts(B, A) & !Trusts(C, B) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))

    model.add_rule(Rule("1.0: Knows(A, B) & Prior('0') -> Trusts(A, B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> Prior('0') ^2"))

    if inv :
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, A) & Trusts(A, B) -> !Trusts(B, A) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, A) & !Trusts(A, B) -> Trusts(B, A) ^2"))

def personality_rules(model) :
    model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> TrustWorthy(B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> Trusting(A) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & !Trusts(A, B) -> !TrustWorthy(B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & !Trusts(A, B) -> !Trusting(A) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Trusting(A) & TrustWorthy(B) -> Trusts(A, B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & !Trusting(A) & !TrustWorthy(B) -> !Trusts(A, B) ^2"))

    model.add_rule(Rule("1.0: Knows(A, B) & Prior('0') -> Trusts(A, B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> Prior('0') ^2"))

def infer(model, datasplit, model_name):
    add_eval_data(model, datasplit, model_name)
    return model.infer(additional_cli_optons = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)

if (__name__ == '__main__'):
    main()
