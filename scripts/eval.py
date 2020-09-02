from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import auc
from sklearn import metrics

import collections
import numpy as np
file_true = open("/Users/niharika/Desktop/LINQS/Trust-prediction/data/trust-prediction/0/eval/trusts_truth.txt", "r+")
file_pred = open("/Users/niharika/Desktop/LINQS/Trust-prediction/cli/inferred-predicates/TRUSTS.txt", "r+")
y_true_lines = file_true.readlines()
y_pred_lines = file_pred.readlines()

# context managers in python
y_true = [] # y_obs
y_pred = []

for line in y_true_lines :
    trustee , trusting, value = line.split()
    y_true.append([trustee, trusting, value])

for line in y_pred_lines :
    trustee , trusting, value = line.split()
    y_pred.append([trustee, trusting, value])

true_out = []
pred_out = []

# a -> b = val
print(len(y_true), len(y_pred) , type(y_true[0][2] ) )
for trustee, trusting, value in y_true :
    true_out.append(value) # int(float(value)*1000)
    for pred_trustee, pred_trusting, pred_val in y_pred :
        if pred_trustee == trustee and pred_trusting == trusting :
            pred_out.append(pred_val) #int(float(pred_relation[2])*1000)

true_out = np.array(true_out, dtype=float)
pred_out = np.array(pred_out, dtype=float)
print(type(true_out[0]))
print(true_out)
print(pred_out)

print("mae", metrics.mean_absolute_error(true_out, pred_out ))
precision, recall, thresholds = precision_recall_curve(true_out, pred_out)

precision_recall = []
for i in range(len(precision)) :
    precision_recall.append((precision[i],recall[i]))
precision_recall.sort(key = lambda x : x[1])
precis = []
recal = []

for i,j in precision_recall :
    precis.append(i)
    recal.append(j)

print(len(precis) , len(recal))
print("auc", auc(recal, precis))
