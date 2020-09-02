from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import auc
from sklearn import metrics

import collections
import numpy as np
file_true = open("/Users/niharika/Desktop/LINQS/Trust-prediction/data/trust-prediction/0/eval/trusts_truth.txt", "r+")
file_pred = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_balance/inferred-predicates/TRUSTS.txt", "r+")
y_true_lines = file_true.readlines()
y_pred_lines = file_pred.readlines()

file_pred_bal5 = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_balance5/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines5 = file_pred_bal5.readlines()

file_pred_status = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_status/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines_status = file_pred_status.readlines()

file_pred_status = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_status_inv/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines_status_inv  = file_pred_status.readlines()

def readfile(y_true_lines, y_pred_lines) :
    y_true = [] # y_obs
    y_pred = []
    for line in y_true_lines :
        trustee , trusting, value = line.split()
        y_true.append([trustee, trusting, value])

    for line in y_pred_lines :
        trustee , trusting, value = line.split()
        y_pred.append([trustee, trusting, value])
    # a -> b = val
    # print(len(y_true), len(y_pred) , type(y_true[0] ) )
    true_out = []
    pred_out = []

    for trustee, trusting, value in y_true :
        true_out.append(value) # int(float(value)*1000)
        for pred_trustee, pred_trusting, pred_val in y_pred :
            if pred_trustee == trustee and pred_trusting == trusting :
                pred_out.append(pred_val) #int(float(pred_relation[2])*1000)

    true_out = np.array(true_out, dtype=float)
    pred_out = np.array(pred_out, dtype=float)
    # print(type(true_out[0]))
    # print(true_out)
    # print(pred_out)
    return (true_out, pred_out)

def maeCalc(observed, predicted):
     mae = metrics.mean_absolute_error(observed, predicted)
     return mae

def auprCalc(observed, predicted):
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
    # print(len(precis) , len(recal))
    aupr =  auc(recal, precis)
    return aupr

print("Results for PSL-BALANCE Model with 16 rules, priors and trust reciprocity.")
obsArr, predArr = readfile(y_true_lines, y_pred_lines)
# print(obsArr, predArr)
psl_balance_mae = maeCalc(obsArr, predArr)
psl_balance_aupr = auprCalc(obsArr, predArr)
print("MAE: ", psl_balance_mae)
print("AUPR: ", psl_balance_aupr)

print("Results for PSL-BALANCE Model with 5 rules, priors and trust reciprocity.")
obsArr5, predArr5 = readfile( y_true_lines, y_pred_lines5 )
# print(obsArr5, predArr5)
psl_balance_mae5 = maeCalc(obsArr5, predArr5)
psl_balance_aupr5 = auprCalc(obsArr5, predArr5)
print("MAE: ", psl_balance_mae5)
print("AUPR: ", psl_balance_aupr5)

print("Results for PSL-Status Model with 8 rules, priors.")
obsStat, predStat = readfile( y_true_lines, y_pred_lines_status )
# print(obsArr5, predArr5)
psl_status_mae = maeCalc(obsStat, predStat)
psl_status_aupr = auprCalc(obsStat, predStat)
print("MAE: ", psl_status_mae)
print("AUPR: ", psl_status_aupr)

print("Results for PSL-Status Model with 8 rules, priors.")
obsStatInv, predStatInv = readfile( y_true_lines, y_pred_lines_status_inv )
# print(obsArr5, predArr5)
psl_status_inv_mae = maeCalc(obsStatInv, predStatInv)
psl_status_inv_aupr = auprCalc(obsStatInv, predStatInv)
print("MAE: ", psl_status_inv_mae)
print("AUPR: ", psl_status_inv_aupr)
