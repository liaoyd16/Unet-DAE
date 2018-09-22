# eval.py

'''
1. 
2. lin. classify task
'''

import os
import utils
import numpy as np
import matplotlib.pyplot as plt
import json

import meta
from meta import *

def Corr(array1, array2):
    array1 = array1.reshape((array1.shape[0] * array1.shape[1],))
    array2 = array2.reshape((array2.shape[0] * array2.shape[1],))
    return np.correlate(array1, array2)


def evalCorrAMI(root):
    # Axx.json
    # Bxx.json
    # AX_BX_clip
    # AX_BX_rec_a
    # AX_BX_rec_b

    namelist = utils.getNames(root)
    a_array = utils.json2numpy( root + "/" + namelist[0])
    a_rec = utils.json2numpy( root + "/" + namelist[3])
    b_array = utils.json2numpy( root + "/" + namelist[1])
    b_rec = utils.json2numpy( root + "/" + namelist[4])

    Corr_a_ap, Corr_b_ap, Corr_a_bp, Corr_b_bp = \
    Corr(a_array, a_rec), Corr(b_array, a_rec), Corr(a_array, b_rec), Corr(b_array, b_rec)

    AMI = Corr_a_ap - Corr_b_ap - Corr_a_bp + Corr_b_bp

    return Corr_a_ap, Corr_b_ap, Corr_a_bp, Corr_b_bp, AMI, Corr_a_ap - Corr_b_ap, Corr_b_bp - Corr_a_bp


def distr(raw_list, list_min, list_max, buckets, fig_name):

    list_bucket = np.zeros(buckets, dtype=int)
    # 5 ranges, 6 nodes
    step = (list_max - list_min) // buckets
    for val in raw_list:
        rang = min( int((val-list_min)//step), buckets-1 )
        list_bucket[rang] += 1

    print(raw_list, np.arange(list_min, list_max, step)[:buckets], list_bucket)

    plt.bar(np.arange(list_min, list_max, step)[:buckets] + step/2, list_bucket, width=step*.75)
    plt.title(fig_name)

    plt.savefig(fig_name+".jpg")
    plt.show()

def displayAMI(jsonname):
    dic = json.load(open(jsonname, "r"))

    ami_list = []
    d_corr_a_list = []
    d_corr_b_list = []
    for key in dic.keys():
        ami_list.append(dic[key][4])
        d_corr_a_list.append(dic[key][5])
        d_corr_b_list.append(dic[key][6])
    ami_list.sort()
    d_corr_a_list.sort()
    d_corr_b_list.sort()

    distr(ami_list, min(ami_list), max(ami_list), 5, "AMI")
    distr(d_corr_a_list, min(d_corr_a_list), max(d_corr_a_list), 5, "A bias")
    distr(d_corr_b_list, min(d_corr_b_list), max(d_corr_b_list), 5, "B bias")


if __name__=="__main__":
    dirlist = os.listdir("result")
    dic = {}
    for d in dirlist:
        if 'A' not in d:
            continue
        print(d)
        Corr_a_ap, Corr_b_ap, Corr_a_bp, Corr_b_bp, AMI, d_Corr_ap, d_Corr_bp = evalCorrAMI(result + d)
        dic[d] = [Corr_a_ap[0], Corr_b_ap[0], Corr_a_bp[0], Corr_b_bp[0], AMI[0], d_Corr_ap[0], d_Corr_bp[0]]

    fhand = open("result/scoring.json", "w")
    fhand.write(json.dumps(dic))
    fhand.close()

    displayAMI(result+"scoring.json")
