#!/usr/bin/env python
#
#  THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
#
#  Copyright (C) 2013
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#  Authors: Tobias Kuehnl <tkuehnl@cor-lab.uni-bielefeld.de>
#           Jannik Fritsch <jannik.fritsch@honda-ri.de>
#

import numpy as np

def evalExp(gtBin, cur_prob, thres, validMap = None):
    '''
    Does the basic pixel based evaluation!
    :param gtBin:
    :param cur_prob:
    :param thres:
    :param validMap:
    '''

    assert len(cur_prob.shape) == 2, 'Wrong size of input prob map'
    assert len(gtBin.shape) == 2, 'Wrong size of input prob map'
    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))

    # histogram of false negatives
    if validMap!=None:
        fnArray = cur_prob[(gtBin == True) & (validMap ==1)]
    else:
        fnArray = cur_prob[(gtBin == True)]
    fnHist = np.histogram(fnArray,bins=thresInf)[0]
    fnCum = np.cumsum(fnHist)
    FN = fnCum[0:0+len(thres)];
    
    if validMap!=None:
        fpArray = cur_prob[(gtBin == False) & (validMap ==1)]
    else:
        fpArray = cur_prob[(gtBin == False)]
    
    fpHist  = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1+len(thres)]

    # count labels and protos
    #posNum = fnArray.shape[0]
    #negNum = fpArray.shape[0]
    if validMap!=None:
        posNum = np.sum((gtBin == True) & (validMap ==1))
        negNum = np.sum((gtBin == False) & (validMap ==1))
    else:
        posNum = np.sum(gtBin == True)
        negNum = np.sum(gtBin == False)
    return FN, FP, posNum, negNum

def pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = None):
    '''

    @param totalPosNum: scalar
    @param totalNegNum: scalar
    @param totalFN: vector
    @param totalFP: vector
    @param thresh: vector
    '''

    #Calc missing stuff
    totalTP = totalPosNum - totalFN
    totalTN = totalNegNum - totalFP


    valid = (totalTP>=0) & (totalTN>=0)
    assert valid.all(), 'Detected invalid elements in eval'

    recall = totalTP / float( totalPosNum )
    precision =  totalTP / (totalTP + totalFP + 1e-10)
    

    #Pascal VOC average precision
    AvgPrec = 0
    for i in np.arange(0,1.1,0.1):
        ind = np.where(recall>=i)
        pmax = max(precision[ind])
        AvgPrec += pmax
    AvgPrec = AvgPrec/11.0
    
    
    # F-measure
    #operation point
    beta = 1.0
    betasq = beta**2
    F = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    index = F.argmax()
    #original threshold
    #index = 500
    MaxF= F[index]
    
    
    recall_bst = recall[index]
    precision_bst =  precision[index]

    TP = totalTP[index]
    TN = totalTN[index]
    FP = totalFP[index]
    FN = totalFN[index]
    valuesMaxF = np.zeros((1,4),'u4')
    valuesMaxF[0,0] = TP
    valuesMaxF[0,1] = TN
    valuesMaxF[0,2] = FP
    valuesMaxF[0,3] = FN

    #ACC = (totalTP+ totalTN)/(totalPosNum+totalNegNum)
    prob_eval_scores  = calcEvalMeasures(valuesMaxF)
    prob_eval_scores['AvgPrec'] = AvgPrec
    prob_eval_scores['MaxF'] = MaxF

    #prob_eval_scores['totalFN'] = totalFN
    #prob_eval_scores['totalFP'] = totalFP
    prob_eval_scores['totalPosNum'] = totalPosNum
    prob_eval_scores['totalNegNum'] = totalNegNum

    prob_eval_scores['precision'] = precision
    prob_eval_scores['recall'] = recall
    #prob_eval_scores['precision_bst'] = precision_bst
    #prob_eval_scores['recall_bst'] = recall_bst
    prob_eval_scores['thresh'] = thresh
    if thresh != None:
        BestThresh= thresh[index]
        prob_eval_scores['BestThresh'] = BestThresh

    #return a dict
    return prob_eval_scores



def calcEvalMeasures(evalDict, tag  = '_wp'):
    '''
    
    :param evalDict:
    :param tag:
    '''
    # array mode!
    TP = evalDict[:,0].astype('f4')
    TN = evalDict[:,1].astype('f4')
    FP = evalDict[:,2].astype('f4')
    FN = evalDict[:,3].astype('f4')
    Q = TP / (TP + FP + FN)
    P = TP + FN
    N = TN + FP
    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    TNR = TN / N
    A = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / P
    #numSamples = TP + TN + FP + FN
    correct_rate = A

    # F-measure
    #beta = 1.0
    #betasq = beta**2
    #F_max = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    
    
    outDict =dict()

    outDict['TP'+ tag] = TP
    outDict['FP'+ tag] = FP
    outDict['FN'+ tag] = FN
    outDict['TN'+ tag] = TN
    outDict['Q'+ tag] = Q
    outDict['A'+ tag] = A
    outDict['TPR'+ tag] = TPR
    outDict['FPR'+ tag] = FPR
    outDict['FNR'+ tag] = FNR
    outDict['PRE'+ tag] = precision
    outDict['REC'+ tag] = recall
    outDict['correct_rate'+ tag] = correct_rate
    return outDict

