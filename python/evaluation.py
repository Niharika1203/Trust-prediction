#!/usr/bin/env python3

from sklearn import metrics
from scipy import stats
import collections
import numpy as np
import os
import csv

DATA_DIR = os.path.join('..', 'data/' )

SPLITS = 8
num_eval_params = 6 # MAE, AUROC, AUPR, AUPR-, spearman Coeff, kendall tau coeff

datasets = [ ("trust-prediction/", True) ,  ("film-trust/", True), ("trust-prediction/", False) ,  ("film-trust/", False) ]
def main():
    for dataset ,square in datasets :
        evaluation_dict = {}

        for data_fold in range(SPLITS) :
            models = [ "balance5", "balance5_recip", "balance_extended", "balance_extended_recip",
              "status" , "status_inv" , "personality", "cyclic_balanced" , "cyclic_bal_unbal" , "similarity",
              "triad-similarity", "triad-personality", "personality-similarity", "triad-pers-sim" ]

            for model_name in models :
                if dataset == "trust-prediction/" and model_name in set(["similarity", "triad-similarity", "personality-similarity", "triad-pers-sim"]) :
                    continue
                outList = evalute( str(data_fold) , model_name, dataset)
                if model_name not in evaluation_dict :
                    evaluation_dict[model_name] = [0] * num_eval_params
                for i in range(num_eval_params) :
                    if outList[i] == "N/A" :
                        evaluation_dict[model_name][i]  = outList[i]
                    else :
                        evaluation_dict[model_name][i] += outList[i]

        dataset_direc = os.path.join(dataset)

        for model, lst in evaluation_dict.items() :
            for i in range(num_eval_params) :
                if evaluation_dict[model][i] != "N/A" :
                    evaluation_dict[model][i] /= SPLITS

        final_output = open( dataset_direc + "squared_" + str(square) +"result.csv", "w+")
        fieldnames = [ 'Model Name', 'Average MAE', 'Average AUROC', 'Average AUPR (positive class)','Average AUPR (negative class)', 'Average Spearman Coeff (Rho)', 'Average Kendall Tau Coefficient' ]
        writer = csv.writer(final_output)
        writer.writerow(fieldnames)

        for model, out in evaluation_dict.items() :
            row = [model] + out
            writer.writerow(row)

def evalute(data_fold, model_name, dataset, square):
    inferred_direc = "/"+ str(data_fold) + "/inferred-predicates/TRUSTS.txt"
    models_direc = os.path.join(dataset,"squared_"+str(square), model_name)
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
        discrete_true_out = []

        for trustee, trusting, value in y_true :
            if float(value) < 0.5 :
                true_out.append(float(value))
                discrete_true_out.append(float(0))
            else :
                discrete_true_out.append(float(1))
                true_out.append(float(value))
            for pred_trustee, pred_trusting, pred_val in y_pred :
                if pred_trustee == trustee and pred_trusting == trusting :
                    pred_out.append(pred_val)


        true_out = np.array(true_out, dtype=float)
        pred_out = np.array(pred_out, dtype=float)

        discrete_true_out = np.array(discrete_true_out, dtype=float)

        return (true_out, pred_out, discrete_true_out)

    obsArr, predArr , discrete_obs_arr = readfile(y_true_lines, y_pred_lines)
    psl_mae = metrics.mean_absolute_error(obsArr, predArr)

    obs_pred = []
    discrete_obs_pred = []
    for i in range(len(obsArr)) :
        obs_pred.append((obsArr[i],predArr[i]))
        discrete_obs_pred.append((discrete_obs_arr[i], predArr[i]))

    obs_pred.sort(key = lambda x : x[0])
    discrete_obs_pred.sort(key = lambda x : x[0])

    observed_arr = []
    predicted_arr = []
    neg_observed_arr = []
    neg_predicted_arr = []

    dis_observed_arr = []
    dis_predicted_arr = []
    dis_neg_observed_arr = []

    for i,j in obs_pred :
        observed_arr.append(i)
        predicted_arr.append(j)
        neg_observed_arr.append(1-i)
        neg_predicted_arr.append(1-j)
    for i,j in discrete_obs_pred:
        dis_observed_arr.append(i)
        dis_predicted_arr.append(j)
        dis_neg_observed_arr.append(1-i)

    psl_auroc = metrics.auc(observed_arr, predicted_arr)
    positiveAUPRC = metrics.average_precision_score(dis_observed_arr, predicted_arr)
    negativeAUPRC = metrics.average_precision_score(dis_neg_observed_arr, neg_predicted_arr)

    correlation, rank = stats.spearmanr(observed_arr, predicted_arr)
    coef, p = stats.kendalltau(observed_arr, predicted_arr)

    evalDir = models_direc +"/" + str(data_fold)
    os.makedirs(evalDir, exist_ok = True)
    eval_file = open(evalDir + "/evaluation_result.csv", "w+")
    eval_file.write("Results for "+ model_name + "\n")

    fieldnames = [ 'Model Name', 'Average MAE', 'Average AUROC', 'Average AUPR (positive class)','Average AUPR (negative class)', 'Average Spearman Coeff (Rho)', 'Average Kendall Tau Coefficient' ]
    csv_writer = csv.writer(eval_file)
    csv_writer.writerow(fieldnames)
    evalList = [psl_mae] + [psl_auroc] + [positiveAUPRC] + [negativeAUPRC] + [correlation] + [coef]
    csv_writer.writerow([model_name] + evalList)

    return evalList

if (__name__ == '__main__') :
    main()
