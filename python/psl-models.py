#!/usr/bin/env python3

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import auc
from sklearn import metrics
from scipy import stats
from scipy.stats import kendalltau
import collections
import numpy as np
import os
from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from pathlib import Path


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'DEBUG'
}

DATA_DIR = os.path.join('..', 'data/' )

SPLITS = 8
ADDITIONAL_CLI_OPTIONS = [
     '--postgres'
]
# "film-trust/", "trust-prediction/"
datasets = [ "trust-prediction/"]
def main():
    for dataset in datasets :
        evaluation_dict = {}

        for data_fold in range(SPLITS) :
            # models = ["similarity"]
            models = [ "balance5", "balance5_recip", "balance_extended", "balance_extended_recip",
              "status" , "status_inv" , "personality", "cyclic_balanced" , "cyclic_bal_unbal" , "similarity",
              "triad-similarity", "triad-personality", "personality-similarity", "triad-pers-sim" ]

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
                elif model_name == "similarity" and dataset == "film-trust/" :
                    similarity(model)
                elif model_name == "triad-similarity" and dataset == "film-trust/" :
                    balance5rules(model)
                    similarity(model, combination = True )
                elif model_name == "triad-personality" and dataset == "film-trust/" :
                    balance5rules(model)
                    personality_rules(model, combination = True)
                elif model_name == "personality-similarity" and dataset == "film-trust/" :
                    similarity(model, combination = True)
                    personality_rules(model)
                elif model_name == "triad-pers-sim" and dataset == "film-trust/" :
                    balance5rules(model)
                    similarity(model, combination = True)
                    personality_rules(model, combination = True)
                else :
                    continue


                print('Rules defined:')
                for rule in model.get_rules():
                    print('   ' + str(rule))

                # Weight Learning
                model = learn(model, str(data_fold) , model_name, dataset)

                print('Learned Rules:')
                for rule in model.get_rules():
                    print('   ' + str(rule))

                # Inference
                results = infer(model, str(data_fold) , model_name, dataset)

                write_results(results, model, model_name, str(data_fold), dataset)

                outList = evalute(model, str(data_fold) , model_name, dataset)
                if model_name not in evaluation_dict :
                    evaluation_dict[model_name] = [0] * 4
                for i in range(4) :
                    if outList[i] == "N/A" :
                        evaluation_dict[model_name][i]  = outList[i]
                    else :
                        evaluation_dict[model_name][i] += outList[i]

        dataset_direc = os.path.join(dataset)
        final_output = open( dataset_direc + "result.txt", "w+")
        for model, lst in evaluation_dict.items() :
            for i in range(4) :
                if evaluation_dict[model][i] != "N/A" :
                    evaluation_dict[model][i] /= SPLITS

        for model, out in evaluation_dict.items() :
            final_output.write(model + "\n")
            final_output.write("Average MAE: " + str(out[0]) + "\n")
            final_output.write("Average AUPR: " + str(out[1]) + "\n")
            final_output.write("Average Rho: " + str(out[2]) + "\n")
            final_output.write("Average Tau: " + str(out[3]) + "\n")


def makeModel(model_name, addPrior = True, square = True, sim = False):
    model = Model(model_name)
    Trusts = Predicate("Trusts", size=2, closed=False)
    Knows = Predicate("Knows", size=2, closed=True)
    Prior = Predicate("Prior", size=1, closed=True)
    model.add_predicate(Trusts)
    model.add_predicate(Knows)
    model.add_predicate(Prior)

    if model_name in ["triad-personality", "personality-similarity", "triad-pers-sim", "personality"] :
        Trusting = Predicate("Trusting", size = 1, closed=False)
        TrustWorthy = Predicate("TrustWorthy", size=1, closed = False)
        model.add_predicate(Trusting)
        model.add_predicate(TrustWorthy)

    if model_name in [ "similarity", "triad-similarity", "personality-similarity", "triad-pers-sim"] :
        SameTastes = Predicate("SameTastes", size = 2, closed = True)
        model.add_predicate(SameTastes)

    return model

def add_learn_data(model, data_fold, model_name, dataset):
    _add_data('learn', model, data_fold, model_name, dataset)

def add_eval_data(model, data_fold, model_name, dataset):
    _add_data('eval', model, data_fold, model_name, dataset )

def _add_data(split, model, data_fold, model_name, dataset):
    split_data_dir = os.path.join(DATA_DIR,dataset, data_fold, split)
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

    if model_name in ["triad-personality", "personality-similarity", "triad-pers-sim", "personality"]:
        path = os.path.join(split_data_dir, 'trusting.txt')
        model.get_predicate('Trusting').add_data_file(Partition.TARGETS, path)
        path = os.path.join(split_data_dir, 'trustworthy.txt')
        model.get_predicate('TrustWorthy').add_data_file(Partition.TARGETS, path)

    if model_name in [ "similarity", "triad-similarity", "personality-similarity", "triad-pers-sim"] :
        path = os.path.join(split_data_dir, 'SameTastes.txt')
        model.get_predicate('SameTastes').add_data_file(Partition.TARGETS, path)

def learn(model, data_fold, model_name, dataset):
    add_learn_data(model, data_fold, model_name, dataset)
    model.learn(additional_cli_options = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)
    return model

def write_results(results, model, model_name, data_fold, dataset):
    out_dir = dataset + "/" + model_name + "/"+ data_fold + '/inferred-predicates'
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

    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(C, A) & Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(C, A) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(C, A) & !Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(C, A) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(A, C) & Knows(C, B) & Trusts(A,B) & Trusts(A,C) & (A != B) & (B != C) & (A != C) -> Trusts(C,B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(A, C) & Knows(C, B) & Trusts(A,B) & !Trusts(A,C) & (A != B) & (B != C) & (A != C) -> !Trusts(C,B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(A, C) & Knows(C, B) & !Trusts(A,B) & Trusts(A,C) & (A != B) & (B != C) & (A != C) -> !Trusts(C,B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & knows(A, C) & knows(C, B) & !Trusts(A,B) & !Trusts(A,C) & (A != B) & (B != C) & (A != C) -> Trusts(C,B) ^2"))

    if unbalanced :
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(C, A) & Trusts(A, B) & !Trusts(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(C, A) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(C, A) & Trusts(A, B) & Trusts(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(C, A) ^2"))
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

def personality_rules(model , combination = False) :
    model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> TrustWorthy(B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> Trusting(A) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & !Trusts(A, B) -> !TrustWorthy(B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & !Trusts(A, B) -> !Trusting(A) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Trusting(A) & TrustWorthy(B) -> Trusts(A, B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & !Trusting(A) & !TrustWorthy(B) -> !Trusts(A, B) ^2"))

    if not combination:
        model.add_rule(Rule("1.0: Knows(A, B) & Prior('0') -> Trusts(A, B) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> Prior('0') ^2"))

def similarity(model, combination = False) :
    if not combination:
        model.add_rule(Rule("1.0: Knows(A, B) & Prior('0') -> Trusts(A, B) ^2"))
        model.add_rule(Rule("1.0: Knows(A, B) & Trusts(A, B) -> Prior('0') ^2"))

    model.add_rule(Rule("1.0: Knows(A, B) & SameTastes(A, B) & (A != B) -> Trusts(A, B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & !SameTastes(A, B) & (A != B) -> !Trusts(A, B) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & Trusts(A, B) & SameTastes(B, C) & (A != B) & (B != C) & (A != C) -> Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, B) & Knows(B, C) & Knows(A, C) & !Trusts(A, B) & SameTastes(B, C) & (A != B) & (B != C) & (A != C) -> !Trusts(A, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, C) & Knows(A, B) & Knows(B, C) & Trusts(A, C) & SameTastes(A, B) & (A != B) & (B != C) & (A != C) -> Trusts(B, C) ^2"))
    model.add_rule(Rule("1.0: Knows(A, C) & Knows(A, B) & Knows(B, C) & !Trusts(A, C) & SameTastes(A, B) & (A != B) & (B != C) & (A != C) -> !Trusts(B, C) ^2"))

def infer(model, data_fold, model_name, dataset):
    add_eval_data(model, data_fold, model_name, dataset)
    return model.infer(additional_cli_options = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)


def evalute(model, data_fold, model_name, dataset):
    inferred_direc = "/"+ str(data_fold) + "/inferred-predicates/TRUSTS.txt"
    models_direc = os.path.join(dataset, model_name)
    truth_file = open(DATA_DIR + dataset + str(data_fold) + "/eval/trusts_truth.txt", "r+")
    y_true_lines = truth_file.readlines()
    file_pred = open(models_direc + inferred_direc, "r+")
    y_pred_lines = file_pred.readlines()

    def readfile(y_true_lines, y_pred_lines) :
        y_true = []
        y_pred = []
        for line in y_true_lines :
            trustee , trusting, value = line.split()
            y_true.append([trustee, trusting, value])

        for line in y_pred_lines :
            trustee , trusting, value = line.split()
            y_pred.append([trustee, trusting, value])

        true_out = []
        pred_out = []

        for trustee, trusting, value in y_true :
            true_out.append(value)
            for pred_trustee, pred_trusting, pred_val in y_pred :
                if pred_trustee == trustee and pred_trusting == trusting :
                    pred_out.append(pred_val)

        true_out = np.array(true_out, dtype=float)
        pred_out = np.array(pred_out, dtype=float)
        return (true_out, pred_out)

    def maeCalc(observed, predicted):
         mae = metrics.mean_absolute_error(observed, predicted)
         return mae

    def auprCalc(observed, predicted):
        # print(observed, predicted)
        # print(type(observed[0]))
        precision, recall, thresholds = precision_recall_curve(observed, predicted)
        precision_recall = []
        for i in range(len(precision)) :
            precision_recall.append((precision[i],recall[i]))
        precision_recall.sort(key = lambda x : x[1])
        precis = []
        recal = []

        for i,j in precision_recall :
            precis.append(i)
            recal.append(j)

        aupr =  auc(recal, precis)
        return aupr

    evalDir = models_direc +"/" + str(data_fold)
    os.makedirs(evalDir, exist_ok = True)
    eval_file = open(evalDir + "/evaluation_result.txt", "w+")

    eval_file.write("Results for "+ model_name + "\n")
    obsArr, predArr = readfile(y_true_lines, y_pred_lines)
    psl_balance_mae = maeCalc(obsArr, predArr)
    if dataset == "film-trust/" :
        psl_balance_aupr = "N/A"
    else :
        psl_balance_aupr = auprCalc(obsArr, predArr)
    correlation, rank = stats.spearmanr(obsArr, predArr)
    coef, p = kendalltau(obsArr, predArr)
    eval_file.write("MAE: "+ str(psl_balance_mae) + " \n")
    eval_file.write("AUPR: " + str(psl_balance_aupr) + " \n")
    eval_file.write("Spearman Rank Coeff: "+ str(correlation) + " \n")
    eval_file.write("Kendall Tau Coeff: " + str(coef) + " \n")
    eval_file.close()
    evalList = [psl_balance_mae] + [psl_balance_aupr] + [correlation] + [coef]
    return evalList

if (__name__ == '__main__'):
    main()
