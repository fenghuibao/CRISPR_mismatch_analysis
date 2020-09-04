import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import math
import argparse

parser = argparse.ArgumentParser(description='Simulate sgRNA invasion')
parser.add_argument('-i','--input_sequence',type=str,required=True,
                    help='DNA target strand (nontemplate)')
parser.add_argument('-t','--nn_table',type=str,required=True,
                    help='path to file of thermodynamic parameters')
parser.add_argument('-k','--temperature',type=float,default=303.15,
                    help='temperature(K)')
args = parser.parse_args()


# parameters
R = 8.314
T = args.temperature

input_sequence = args.input_sequence
nn_table = pd.read_csv(args.nn_table,sep='\s+')
nn_table['dG'] = nn_table['dH']-T*nn_table['dS']
nn_table.index = nn_table['sequence'].values
nn_table = nn_table.drop('sequence',1)

ddpairs = {'A':'T','T':'A','C':'G','G':'C'} # Watson Crick base pairs for DNA/DNA duplexes
drpairs = {'A':'U','T':'A','C':'G','G':'C'} # Watson Crick base pairs for DNA/RNA duplexes
rdtransform = {'A':'A','C':'C','G':'G','U':'T'}
mutation_type = {'A':'A','C':'C','G':'G','T':'U'}
tDNA_seq = [ddpairs[base] for base in input_sequence]
ntDNA_seq = input_sequence[::-1]
sgRNA_seq = [drpairs[base] for base in tDNA_seq]

def get_nearest_neighbors(sgRNA_seq):
    dd = ['d'+ntDNA_seq[i+1]+ntDNA_seq[i]+'/d'+ddpairs[ntDNA_seq[i+1]]+ddpairs[ntDNA_seq[i]] for i in range(19)]
    rd = ['r'+sgRNA_seq[i+1]+sgRNA_seq[i]+'/d'+ddpairs[ntDNA_seq[i+1]]+ddpairs[ntDNA_seq[i]] for i in range(19)]
    nearest_neighbors = dd+rd
    return nearest_neighbors

def get_states(nearest_neighbors, Cas_dG):
    dG = [nn_table.loc[nn,'dG'] for nn in nearest_neighbors]
    ddG = [dG[i+19]-dG[i+1]-Cas_dG for i in range(18)]
    bz = np.cumsum([-deltaG/(R*T) for deltaG in ddG])
    states = softmax(bz)
    P_off = states[0]
    return -np.log10(P_off)

def main(Cas_dG=3000):
    activity =pd.DataFrame(columns=['-log10P'])
    nearest_neighbors = get_nearest_neighbors(sgRNA_seq=sgRNA_seq[::-1])
    # WT
    activity.loc[input_sequence,'-log10P'] = get_states(nearest_neighbors, Cas_dG)
    # single_mismatch
    for i in range(1,19):
        sgRNA_seq_s = sgRNA_seq[:]
        sgRNA_seq_s[i] = mutation_type[tDNA_seq[i]]
        nearest_neighbors_s = get_nearest_neighbors(sgRNA_seq=sgRNA_seq_s[::-1])
        activity.loc[''.join([rdtransform[base] for base in sgRNA_seq_s]),'-log10P'] = get_states(nearest_neighbors_s, Cas_dG)
    # double_mismatch
    for i in range(1,17):
        for j in range(i+2,19):
            sgRNA_seq_d = sgRNA_seq[:]
            sgRNA_seq_d[i] = mutation_type[tDNA_seq[i]]
            sgRNA_seq_d[j] = mutation_type[tDNA_seq[j]]
            nearest_neighbors_d = get_nearest_neighbors(sgRNA_seq=sgRNA_seq_d[::-1])
            activity.loc[''.join([rdtransform[base] for base in sgRNA_seq_d]),'-log10P'] = get_states(nearest_neighbors_d, Cas_dG)
    activity['normalized_activity'] = (activity['-log10P'] - activity['-log10P'].min()) / (activity['-log10P'].max() - activity['-log10P'].min())
    activity.sort_values(by='normalized_activity', ascending=False, inplace=True)
    activity.to_csv('activity.txt', sep='\t')

if __name__ == '__main__':
    main()
