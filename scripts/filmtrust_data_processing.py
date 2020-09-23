
import pandas as pd
import os

set = set()
folds = 8
# for data_fold in range(folds) :
#     file = pd.read_csv("../data/film-trust/" + str(data_fold) + "/eval/knows_obs.txt", sep ="\t", names = ["U1","U2", "trust"])
#     file2 = pd.read_csv("../data/film-trust/" + str(data_fold) + "/learn/knows_obs.txt", sep ="\t", names = ["U1","U2", "trust"])
#
#     for index, row in file.iterrows() :
#         set.add(row['U1'])
#         set.add(row['U2'])
#
#     for index, row in file2.iterrows() :
#         set.add(row['U1'])
#         set.add(row['U2'])
#
# users_list = sorted(list(set))
# # print(sorted(users_list))
#
# for data_fold in range(folds) :
#
#     users = open("../data/film-trust/" + str(data_fold) + "/eval/trusting.txt", "w")
#     users2 = open("../data/film-trust/" + str(data_fold) + "/learn/trusting.txt", "w")
#
#     users3 = open("../data/film-trust/" + str(data_fold) + "/eval/trustworthy.txt", "w")
#     users4 = open("../data/film-trust/" + str(data_fold) + "/learn/trustworthy.txt", "w")
#
#     for i in range(len(users_list)) :
#         users.write(str(int(users_list[i])))
#         users.write("\n")
#         users2.write(str(int(users_list[i])))
#         users2.write("\n")
#         users3.write(str(int(users_list[i])))
#         users3.write("\n")
#         users4.write(str(int(users_list[i])))
#         users4.write("\n")

for data_fold in range(folds) :
    for split in ["eval", "learn"] :
        users = open("../data/film-trust/" + str(data_fold) + "/" + split + "/trusts_target.txt", "w")
        file = pd.read_csv("../data/film-trust/" + str(data_fold) + "/" + split +"/trusts_truth.txt", sep ="\t", names = ["U1","U2", "trust"])

        for index, row in file.iterrows() :
            users.write(str(int(row['U1'])) + "\t" + str(int(row['U2'])) )
            users.write("\n")
