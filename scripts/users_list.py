
import pandas as pd
import os

file = pd.read_csv("../data/trust-prediction/0/eval/knows_obs.txt", sep ="\t", names = ["U1","U2", "trust"])
users = open("../data/trust-prediction/0/eval/users.txt", "w")
users2 = open("../data/trust-prediction/0/learn/users_learn.txt", "w")

users3 = open("../data/trust-prediction/0/eval/usersTW.txt", "w")
users4 = open("../data/trust-prediction/0/learn/users_learnTW.txt", "w")

set = set()
for index, row in file.iterrows() :
    set.add(row['U1'])
    set.add(row['U2'])

users_list = (list(set))
for i in range(len(users_list)) :
    users.write(str(int(users_list[i])))
    users.write("\n")
    users2.write(str(int(users_list[i])))
    users2.write("\n")
    users3.write(str(int(users_list[i])))
    users3.write("\n")
    users4.write(str(int(users_list[i])))
    users4.write("\n")
