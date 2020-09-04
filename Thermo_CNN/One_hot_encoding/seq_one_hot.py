import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os
import argparse
import pickle


#library r0
sgRNA = 'UACACUUGAACUACCGCGAAG'
comp_d = 'ATGTGAACTTGATGGCGCTTC'
noncomp_d = 'CTACACTTGAACTACCGCGAG'
data = pd.read_csv('regression_data_r0.txt', sep='\s+')

#library r349
'''
sgRNA = 'AGUGUCAUUGUUGAAUUCUUA'
comp_d = 'TCACAGTAACAACTTAAGAAT'
noncomp_d = 'CAGTGTCATTGTTGAATTCTA'
data = pd.read_csv('regression_data_r349.txt', sep='\s+')
'''


RNA_lst = ['A', 'C', 'G', 'U']
DNA_lst = ['A', 'C', 'G', 'T']
# kernel of nearest neighbor
def get_nearest_neighbor():
    nn_lst = ['rAA/dAA-dAT/dTA']
    layer = tf.one_hot(indices=[[[0, 0], [0, 0], [0, 3]]], depth=4, dtype=tf.float64)
    m = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    for n in range(4):
                        if m:
                            conv = tf.one_hot(indices=[[[i, j], [k, l], [n, 3 - k]]], depth=4, dtype=tf.float64)
                            layer = tf.concat([layer, conv], axis=0)
                            nn_lst.append('r'+RNA_lst[i]+RNA_lst[j] +'/d'+DNA_lst[k]+DNA_lst[l]+'-d'+DNA_lst[n]+DNA_lst[3-k]+'/d'+DNA_lst[3-n]+DNA_lst[k])
                        else:
                            m = 1
    return tf.reshape(layer, [-1, 3, 2, 4, 1]), nn_lst

# kernel of initial site
def get_initial_state():
    initial_lst = ['rAA/dAA/dTT']
    layer = tf.one_hot(indices=[[[0, 0], [0, 0], [3, 3]]], depth=4, dtype=tf.float64)
    m = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    if m:
                        conv = tf.one_hot(indices=[[[i, j], [k, l], [3 - k, 3 - l]]], depth=4, dtype=tf.float64)
                        layer = tf.concat([layer, conv], axis=0)
                        initial_lst.append('r'+RNA_lst[i]+RNA_lst[j] +'/d'+DNA_lst[k]+DNA_lst[l]+'/d'+DNA_lst[3-k]+DNA_lst[3-l])
                    else:
                        m = 1
    return tf.reshape(layer, [-1, 3, 2, 4, 1]), initial_lst


nearest_neighbor, nn_lst = get_nearest_neighbor()
initial_seq, initial_lst = get_initial_state()


def base_one_hot(base):
    if base == 'A':
        return [1., 0., 0., 0.]
    elif base == 'C':
        return [0., 1., 0., 0.]
    elif base == 'G':
        return [0., 0., 1., 0.]
    else:
        return [0., 0., 0., 1.]

def one_hot_encoding(sgRNA, comp_d, noncomp_d):
    seq_sgRNA = [base_one_hot(base) for base in sgRNA]
    seq_comp_d = [base_one_hot(base) for base in comp_d]
    seq_noncomp_d = [base_one_hot(base) for base in noncomp_d]
    return [seq_sgRNA, seq_comp_d, seq_noncomp_d]

seq = [one_hot_encoding(sgRNA, comp_d, noncomp_d)]
score = [[[0.]]]
sgRNA_ = list('UACACUUGAACUACCGCGAG')
base_pair = {'A':'T','C':'G','U':'A','G':'C'}
for i in data.index:
    label = data.columns[(data.loc[i] == 1) | (data.loc[i] == 2)].values
    if len(label) == 2:
        MT_site, MT_type = int(label[0][2:])-1, label[1][-1]
        sgRNA_s = sgRNA_[:]
        sgRNA_s[MT_site] = MT_type
        sgRNA_s = sgRNA_s[:-1] + sgRNA_s[-2:]
        seq.append(one_hot_encoding(sgRNA_s, comp_d, noncomp_d))
    elif len(label) == 3:
        MT_site1, MT_site2, MT_type = int(label[0][2:])-1, int(label[1][2:])-1, label[2][-1]
        sgRNA_d = sgRNA_[:]
        sgRNA_d[MT_site1] = MT_type
        sgRNA_d[MT_site2] = MT_type
        sgRNA_d = sgRNA_d[:-1] + sgRNA_d[-2:]
        seq.append(one_hot_encoding(sgRNA_d, comp_d, noncomp_d))
    else:
        MT_site1, MT_site2, MT_type1, MT_type2 = int(label[0][2:])-1, int(label[1][2:])-1, label[2][-1], label[3][-1]
        base1, base2 = label[2][-3], label[3][-3]
        sgRNA_d = sgRNA_[:]
        if base1 == base_pair[sgRNA_[MT_site1]]:
            sgRNA_d[MT_site1] = MT_type1
            sgRNA_d[MT_site2] = MT_type2
        elif base1 == base_pair[sgRNA_[MT_site2]]:
            sgRNA_d[MT_site1] = MT_type2
            sgRNA_d[MT_site2] = MT_type1
        else:
            print(1)
        sgRNA_d = sgRNA_d[:-1] + sgRNA_d[-2:]
        seq.append(one_hot_encoding(sgRNA_d, comp_d, noncomp_d))
    score.append([[data.loc[i,'score']]])

seq_ = tf.cast(tf.constant(seq), tf.float64)
nn = tf.concat([tf.nn.conv2d(seq_[:, :, :-1, :], conv, strides=[1, 1, 1, 1], padding='VALID') for conv in nearest_neighbor], axis=-1)
nn = tf.nn.relu(nn - 5)
nn_ = tf.squeeze(nn).numpy()
indices = np.argwhere(nn_==1)
nn_num = {n:0 for n in nn_lst}
for i in indices:
    nn_num[nn_lst[i[-1]]] += 1
a = pd.Series(nn_num)
nn_trained = list(a[a>2].keys())
with open('nn.pickle', 'wb') as f:
    pickle.dump(nn_trained, f)


train_seq, test_seq, train_score, test_score = train_test_split(seq, score, test_size=0.2, random_state=10)
with open('seq_train.pickle', 'wb') as f:
    pickle.dump(train_seq, f)
with open('score_train.pickle', 'wb') as f:
    pickle.dump(train_score, f)
with open('seq_test.pickle', 'wb') as f:
    pickle.dump(test_seq, f)
with open('score_test.pickle', 'wb') as f:
    pickle.dump(test_score, f)