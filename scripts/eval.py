from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import auc
from sklearn import metrics
from scipy import stats
from scipy.stats import kendalltau

import collections
import numpy as np
file_true = open("/Users/niharika/Desktop/LINQS/Trust-prediction/data/trust-prediction/0/eval/trusts_truth.txt", "r+")
file_pred = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_balance/inferred-predicates/TRUSTS.txt", "r+")
y_true_lines = file_true.readlines()
y_pred_lines = file_pred.readlines()

file_pred_bal5 = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_balance5/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines5 = file_pred_bal5.readlines()

file_pred_recip = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_balance_recip/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines_recip = file_pred_recip.readlines()

file_pred_bal5_recip = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_balance5_recip/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines5_recip = file_pred_bal5_recip.readlines()

file_pred_status = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_status/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines_status = file_pred_status.readlines()

file_pred_status_inv = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_status_inv/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines_status_inv  = file_pred_status_inv.readlines()

file_pred_cyclic = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_cyclic_noncyclic/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines_cyclic = file_pred_cyclic.readlines()

file_pred_cyclic_bnb = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_cyclic_noncyclic_b_unb/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines_cyclic_bnb = file_pred_cyclic_bnb.readlines()

file_pred_pers = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_personality /inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines_pers = file_pred_pers.readlines()

file_pred_triad_pers = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli_triad_personality/inferred-predicates/TRUSTS.txt", "r+")
y_pred_lines_triad_pers = file_pred_triad_pers.readlines()

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

print("*********************************************** \n")

print("Results for PSL-BALANCE Model with 16 rules and priors.")
obsArr, predArr = readfile(y_true_lines, y_pred_lines)
# print(obsArr, predArr)
psl_balance_mae = maeCalc(obsArr, predArr)
psl_balance_aupr = auprCalc(obsArr, predArr)
correlation, rank = stats.spearmanr(obsArr, predArr)
coef, p = kendalltau(obsArr, predArr)
print("MAE: ", psl_balance_mae)
print("AUPR: ", psl_balance_aupr)
print("Spearman Rank Coeff", correlation)
print("Kendall Tau Coeff", coef)
print("*********************************************** \n")

print("Results for PSL-BALANCE Model with 16 rules, priors and trust reciprocity.")
obsArr_recip, predArr_recip = readfile(y_true_lines, y_pred_lines_recip)
# print(obsArr, predArr)
psl_balance_recip_mae = maeCalc(obsArr_recip, predArr_recip)
psl_balance_recip_aupr = auprCalc(obsArr_recip, predArr_recip)
correlation, rank = stats.spearmanr( obsArr_recip, predArr_recip )
coef, p = kendalltau( obsArr_recip, predArr_recip )
print("MAE: ", psl_balance_recip_mae)
print("AUPR: ", psl_balance_recip_aupr)
print("Spearman Rank Correlation", correlation )
print("Kendall Tau Coeff", coef)
print("*********************************************** \n")
print("Results for PSL-BALANCE Model with 5 rules, priors.")
obsArr5, predArr5 = readfile( y_true_lines, y_pred_lines5 )
# print(obsArr5, predArr5)
psl_balance_mae5 = maeCalc(obsArr5, predArr5)
psl_balance_aupr5 = auprCalc(obsArr5, predArr5)
correlation, rank = stats.spearmanr( obsArr5, predArr5)
coef, p = kendalltau( obsArr5, predArr5 )
print("MAE: ", psl_balance_mae5)
print("AUPR: ", psl_balance_aupr5)
print("Spearman Rank Correlation", correlation)
print("Kendall Tau Coeff", coef)

print("*********************************************** \n")
print("Results for PSL-BALANCE Model with 5 rules, priors and trust reciprocity.")
obsArr5_recip, predArr5_recip = readfile( y_true_lines, y_pred_lines5_recip )
# print(obsArr5, predArr5)
psl_balance_recip_mae5 = maeCalc(obsArr5_recip, predArr5_recip)
psl_balance_recip_aupr5 = auprCalc(obsArr5_recip, predArr5_recip)
correlation, rank = stats.spearmanr(obsArr5_recip, predArr5_recip )
coef, p = kendalltau( obsArr5_recip, predArr5_recip )
print("MAE: ", psl_balance_recip_mae5)
print("AUPR: ", psl_balance_recip_aupr5)
print("Spearman Rank Correlation", correlation)
print("Kendall Tau Coeff", coef)
print("*********************************************** \n")
print("Results for PSL-Status Model with 8 rules and priors.")
obsStat, predStat = readfile( y_true_lines, y_pred_lines_status )
# print(obsArr5, predArr5)
psl_status_mae = maeCalc(obsStat, predStat)
psl_status_aupr = auprCalc(obsStat, predStat)
correlation, rank = stats.spearmanr( obsStat, predStat )
coef, p = kendalltau( obsStat, predStat )
print("MAE: ", psl_status_mae)
print("AUPR: ", psl_status_aupr)
print("Spearman Rank Correlation", correlation)
print("Kendall Tau Coeff", coef)

print("*********************************************** \n")
print("Results for PSL-Status Model with 8 rules, priors and with Inversion rules.")
obsStatInv, predStatInv = readfile( y_true_lines, y_pred_lines_status_inv )
# print(obsArr5, predArr5)
psl_status_inv_mae = maeCalc(obsStatInv, predStatInv)
psl_status_inv_aupr = auprCalc(obsStatInv, predStatInv)
correlation, rank = stats.spearmanr( obsStatInv, predStatInv )
coef, p = kendalltau( obsStatInv, predStatInv )
print("MAE: ", psl_status_inv_mae)
print("AUPR: ", psl_status_inv_aupr)
print("Spearman Rank Correlation", correlation)
print("Kendall Tau Coeff", coef)
print("*********************************************** \n")

print("Results for PSL-cyclic non-cyclic Model with 6 balanced rules and priors.")
obsArr7, predArr7 = readfile( y_true_lines, y_pred_lines_cyclic )
psl_mae7 = maeCalc(obsArr7, predArr7)
psl_aupr7 = auprCalc(obsArr7, predArr7)
correlation, rank = stats.spearmanr( obsArr7, predArr7)
coef, p = kendalltau( obsArr7, predArr7 )
print("MAE: ", psl_mae7)
print("AUPR: ", psl_aupr7)
print("Spearman Rank Correlation", correlation)
print("Kendall Tau Coeff", coef)
print("*********************************************** \n")
print("Results for PSL-cyclic non-cyclic Model with 12 balanced and unbalanced rules and priors.")
obsArr10, predArr10 = readfile( y_true_lines, y_pred_lines_cyclic_bnb )
psl_mae10 = maeCalc(obsArr10, predArr10)
psl_aupr10 = auprCalc(obsArr10, predArr10)
correlation, rank = stats.spearmanr( obsArr10, predArr10 )
coef, p = kendalltau( obsArr10, predArr10 )
print("MAE: ", psl_mae7)
print("AUPR: ", psl_aupr7)
print("Spearman Rank Correlation", correlation)
print("Kendall Tau Coeff", coef)
print("*********************************************** \n")
print("Results for PSL_personality model.")
obsArr8, predArr8 = readfile( y_true_lines, y_pred_lines_pers )
psl_mae8 = maeCalc(obsArr8, predArr8)
psl_aupr8 = auprCalc(obsArr8, predArr8)
correlation, rank = stats.spearmanr( obsArr8, predArr8 )
coef, p = kendalltau( obsArr8, predArr8 )
print("MAE: ", psl_mae8)
print("AUPR: ", psl_aupr8)
print("Spearman Rank Correlation", correlation)
print("Kendall Tau Coeff", coef)
print("*********************************************** \n")
print("Results for PSL-PSL_personality Model with 16 triadic rules, priors.")
obsArr9, predArr9 = readfile( y_true_lines, y_pred_lines_triad_pers )
psl_mae9 = maeCalc(obsArr9, predArr9)
psl_aupr9 = auprCalc(obsArr9, predArr9)
correlation, rank = stats.spearmanr( obsArr9, predArr9 )
coef, p = kendalltau( obsArr9, predArr9 )
print("MAE: ", psl_mae9)
print("AUPR: ", psl_aupr9)
print("Spearman Rank Correlation", correlation)
print("Kendall Tau Coeff", coef)
print("*********************************************** \n")
