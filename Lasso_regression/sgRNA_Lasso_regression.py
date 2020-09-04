# this script is used to train a gradient boosting regression model for sgRNA activity prediction

from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import logging
import argparse
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def L1_LR_ML_parseargs():
    """
    Parse arguments. Only used when L1_LR_ML.py is executed directly.
    """
    parser=argparse.ArgumentParser(description='Linear Regression to deciper sgRNA activity from sequence feature')
    parser.add_argument('-i', '--dataset', required=True, help='raw dataset of sgRNA feature and score')
    parser.add_argument('-s', '--score', required=True, help='type of used score in this work')
    parser.add_argument('-c', '--cvfolds', type=int, default=5, help='fold of cross-validation in this work')
    parser.add_argument('-a', '--alphas', default=None, help='tested alphas value')
    parser.add_argument('-r', '--random', type=int, default=10, help='random state value')
    args=parser.parse_args()
    return args

# ///////////////////////////////////////////////////////
def modeltest(alg, dtrain, dtarget, predictors, cv_folds, performTest=False, printFeatureImportance=True):
    """
    Test and present the model
    two utilities:
    performTest: given a trained model, test the performance on an unseen dataset
    printFeatureImportance: get the feature importance evaluation
    Parameters:
    ======================
    alg: model
    dtrain: dataset with features
    dtarget: dataset with column of target
    predictors: column names of features
    cv_folds: fold of cross validation
    """
    # test the model by an unseen dataset:
    if performTest:
        # default R^2
        print ('R^2: %.4f'%(alg.score(dtrain, dtarget)))
        # correlation coefficient
        dtrain_predictions = alg.predict(dtrain)
        Cor,p=spearmanr(dtarget, dtrain_predictions)
        print ('Spearman CC: %.4f'%(Cor))
        Cor,p=pearsonr(dtarget, dtrain_predictions)
        print ('Pearson CC: %.4f'%(Cor))
        print ('')
        plt.scatter(np.array(dtrain_predictions), np.array(dtarget))
        plt.xlabel('Model prediction', fontsize='x-large')
        plt.ylabel('Experimental observation', fontsize='x-large')
        XYmax=max(np.max(dtrain_predictions), np.max(dtarget))
        XYmin=min(np.min(dtrain_predictions), np.min(dtarget))
        plt.xlim((XYmin, XYmax))
        plt.ylim((XYmin, XYmax))
        plt.annotate('$R^2$ = %.4f'%(alg.score(dtrain, dtarget)), xy=(0.05,0.9), xycoords='axes fraction', fontsize='x-large')
        plt.savefig('test_prediction.png',dpi=400)
        plt.clf()
    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.coef_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar',color='b',alpha=0.4)
        plt.ylabel('Feature Coefficient',fontsize='x-large')
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.9)
        plt.savefig('baseline_model_feature.png',dpi=400)
        plt.clf()

# //////////////////////////////////////////
def L1_LR_ML_main(args):
    """
    Main entry
    """
    """
    get input parameters
    """
    # random state
    rsv = args.random
    # raw dataset with features and scores
    my_data=pd.read_csv(args.dataset,sep='\s+')
    # scores used in this round
    score_type=args.score
    # fold of cross validation used in this round
    cv_folds=args.cvfolds
    # alpha value
    alphas=args.alphas
    predictors = [x for x in my_data.columns if x!=score_type]
    # split the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(my_data[predictors], my_data[score_type], test_size=0.2, random_state=rsv)
    print ('')
    print ('Number of entries in training set: %d'%(len(y_train)))
    print ('Number of entries in test set: %d'%(len(y_test)))
    print ('')
    # the model with default setting
    print ('Model with default parameter (test dataset performance)')
    # optimization of alpha
    L1LR = LassoCV(alphas=alphas,cv=cv_folds,max_iter=1e5,fit_intercept=False).fit(X_train,y_train)
    log_alphas = -np.log10(L1LR.alphas_)
    plt.plot(log_alphas,L1LR.mse_path_.mean(axis=-1),'k--',label='Average MSE across the folds')
    plt.axvline(-np.log10(L1LR.alpha_),color='k',label=r'$\alpha$: CV estimate')
    print(-np.log10(L1LR.alpha_))
    plt.legend(loc='best')
    plt.xlabel(r'-log$\alpha$', fontsize='x-large')
    plt.ylabel('Mean squared error', fontsize='x-large')
    plt.savefig('optimization of alpha.png',dpi=1000)
    plt.clf()
    modeltest(L1LR, X_test, y_test, predictors, cv_folds, performTest=True)

if __name__ == '__main__':
    try:
        args=L1_LR_ML_parseargs()
        L1_LR_ML_main(args)
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted.\n")
        sys.exit(0)
