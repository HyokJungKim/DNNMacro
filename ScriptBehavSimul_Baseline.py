# -*- coding: utf-8 -*-
"""
ScriptBehavSimul.py
 - Simulate behavioral macro models with DNN
 
Version 1: V = U
"""

# Native Python packages
import os

# Other packages
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.nn import relu
from tensorflow.keras import regularizers

from tqdm import tqdm

from ctypes import CDLL, c_int, c_float, byref

# Read path
WorkPath = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# PART 0: PREPROCESSING
# -----------------------------------------------------------------------------
# Read Fortran DLL: change directory later
#DLLChoice = CDLL('D:/CloudStation/BehavioralMacro/Writings/LiteratureReview/Codes/UpdateChoiceGrid/UpdateChoiceGrid/x64/Debug/UpdateChoiceGrid.dll')
DLLChoice = CDLL('D:/CloudStation/BehavioralMacro/Writings/LiteratureReview/Codes/UpdateChoiceGrid/UpdateChoiceGrid/UpdateChoiceGrid_Cuda.dll')

# Linux
#DLLChoice = CDLL(WorkPath + '/UpdateChoiceGrid.so')

DLLChoice.FortranUpdateChoiceGrid.restype=None

# -----------------------------------------------------------------------------
# PART 0-1) DATA
# -----------------------------------------------------------------------------
# Read data
Data = pd.read_csv(WorkPath+'/DataBehavSimul.csv', header=0, index_col=None)

YY = Data['income'][0:3]            # Income data
YY = np.array(YY, copy=True)    # Convert as numpy array

RR = Data['rate1'][0:3]           # Interest rate data
RR = 1 + np.array(RR, copy=True)    # Convert as numpy array

# -----------------------------------------------------------------------------
# PART 0-2) PARAMETERS
# -----------------------------------------------------------------------------
ssigma = 0.5        # CRRA parameter
bb = 1.0            # Discount factor - beta
dd = 0.98           # Discount factor - delta

Nendo = 2           # Number of endogenous variables c(t), a(t)
NLag = 7            # Lags for input data

TT = YY.shape[0]  # Total stages

NNwidth = 10
NNdepth = 2
NNinput = 2

# -----------------------------------------------------------------------------
# PART 1: DEFINE NEURAL NETWORK
# -----------------------------------------------------------------------------
ListVal = []     # List of value functions

def my_leaky_relu(x):
    in_alpha = 2.0
    return tf.nn.relu(x) - in_alpha*tf.nn.relu(-x)

# Value and policy functions for each stage of life
for tt in range(1,TT):
    # Model 1 (Value function)
    Linput = keras.Input(shape=(NNinput,))    # Input: a(t-1), d(t-1)
    L1 = layers.Dense(NNwidth, activation=my_leaky_relu, use_bias=False,
                      kernel_regularizer=regularizers.l2(0.01))(Linput)
    L1_b  = layers.BatchNormalization(axis=1, epsilon=0.0001)(L1)
    L2 = layers.Dense(NNwidth, activation=relu, use_bias=False,
                      kernel_regularizer=regularizers.l2(0.01))(L1_b)
    L2_b  = layers.BatchNormalization(axis=1, epsilon=0.0001)(L2)
    L3 = layers.Dense(NNwidth, activation=my_leaky_relu, use_bias=False,
                      kernel_regularizer=regularizers.l2(0.01))(L2_b)
    L3_b  = layers.BatchNormalization(axis=1, epsilon=0.0001)(L3)
    Loutput = layers.Dense(1, activation=relu, kernel_regularizer=regularizers.l2(0.01),
                           bias_regularizer=regularizers.l2(0.01))(L3_b)
    # Define model and compile
    tempModel_Val = keras.models.Model(inputs=Linput, outputs=Loutput)
    tempModel_Val.compile(optimizer=tf.train.AdamOptimizer(),loss='mse', metrics=['mae'])

    ListVal.append(tempModel_Val)   # Append value function to list

# -----------------------------------------------------------------------------
# FUNCTIONS USED IN THIS SCRIPT
# -----------------------------------------------------------------------------
# CRRA utility function
def funU(inSigma, inC):
    return (inC**(1-inSigma))/(1-inSigma)

def funReLU(inVec):
    tempBool = inVec >= 0
    return tempBool*inVec

def funLeakyReLU(inVec):
    inAlpha = 2.0
    
    tempBool = inVec >= 0
    return tempBool*inVec + (1-tempBool)*inAlpha*inVec

def funBNTransform(inVec, in_gamma, in_beta, in_mean, in_std):
    return in_gamma*(inVec - in_mean)/in_std + in_beta

# Percentage point choices to values
def funPct2Val(inFunPct, inFunExog):
    outArray = np.zeros((3,))

    inFunC1pct = inFunPct[0]    # Percentage of consumption
    inFunA1pct = inFunPct[1]    # Percentage of asset
    
    inFunY1 = inFunExog[0]
    inFunR1 = inFunExog[1]
    inFunA0 = inFunExog[2]
    inFunD0 = inFunExog[3]
    
    outC1 = (inFunY1+inFunD0)*inFunC1pct
    outA1 = (inFunY1+inFunR1*inFunA0+inFunD0-outC1)*inFunA1pct
    outD1 = (inFunY1+inFunR1*inFunA0+inFunD0-outC1)*(1-inFunA1pct)
    
    outArray[0] = outC1
    outArray[1] = outA1
    outArray[2] = outD1

    return outArray

# Evaluate the values
def funEvalVal(inChoices, inYYseries, inRRseries, inEvalValParam):
    inTT = inEvalValParam[0]
    inbb = inEvalValParam[1]
    indd = inEvalValParam[2]
    inssigma = inEvalValParam[3]
    
    choiceCpct = inChoices[:,0]
    choiceApct = inChoices[:,1]
    
    outChoices = np.zeros((inTT, 3))
    outVt = np.zeros((inTT, 1))
    UU = np.zeros((inTT, 1))
    
    tempA = 0
    tempD = 0
    
    for tt in range(0,inTT):
        Exog = np.array([inYYseries[tt], inRRseries[tt], tempA, tempD])
        temp = np.array([choiceCpct[tt], choiceApct[tt]])
        
        tempChoices = funPct2Val(temp, Exog)
        
        tempA = tempChoices[1]
        tempD = tempChoices[2]
        
        outChoices[tt,:] = tempChoices
        
        UU[tt,:] = funU(inssigma, tempChoices[0])
    
    for tt in range(0,inTT):
        tempV = UU[tt,0]
        
        if tt+1 == inTT:
            pass
        else:
            for qq in range(tt+1,inTT):
                if tt == 0:
                    in_discount = inbb*(indd**(qq-tt))
                else:
                    in_discount = indd**(qq-tt)
                    
                tempV = tempV + in_discount*UU[qq,0]
        
        outVt[tt,:] = tempV
    
    return [outVt, outChoices]

# Calculate V(t+1;State(t))
def funGetOutput(inInput, inWeight):
    outOutput = np.matmul(inInput, inWeight[0])
    outOutput = funLeakyReLU(outOutput)
    
    in_gamma = inWeight[1]
    in_beta  = inWeight[2]
    in_mean  = inWeight[3]
    in_std   = np.sqrt(inWeight[4] + 0.0001)
    
    outOutput = funBNTransform(outOutput, in_gamma, in_beta, in_mean, in_std)
    
    outOutput = np.matmul(outOutput, inWeight[5])
    outOutput = funReLU(outOutput)
    
    in_gamma = inWeight[6]
    in_beta  = inWeight[7]
    in_mean  = inWeight[8]
    in_std   = np.sqrt(inWeight[9] + 0.0001)
    
    outOutput = funBNTransform(outOutput, in_gamma, in_beta, in_mean, in_std)
    
    outOutput = np.matmul(outOutput, inWeight[10])
    outOutput = funLeakyReLU(outOutput)
    
    in_gamma = inWeight[11]
    in_beta  = inWeight[12]
    in_mean  = inWeight[13]
    in_std   = np.sqrt(inWeight[14] + 0.0001)
    
    outOutput = funBNTransform(outOutput, in_gamma, in_beta, in_mean, in_std)
    
    outOutput = np.matmul(outOutput, inWeight[15])
    outOutput = funReLU(outOutput+inWeight[16])
    
    # output is 1 by 1 array so should fix
    return outOutput[0]

def funTrainDNN(inListVal, inXX, inYY, inCut, inTT, inSavePath):
    outWeights = []
    
    for tt in range(0,inTT-1):
        inListVal[tt].fit(inXX[inCut:,:,tt], inYY[inCut:,:,tt],
                        epochs=100, batch_size=50, verbose=0,
                        validation_data=(inXX[0:inCut,:,tt],inYY[0:inCut,:,tt]))
        inListVal[tt].save_weights(inSavePath+str(tt))
        
        outWeights.append(inListVal[tt].get_weights())
        
        print('Age: ', tt+20, 'trained.')
    return [inListVal, outWeights]

# funArraycType: Transform numpy array into one dimensional ctype array
def funArraycType(inMat, incType):
    # inMat: numpy array
    # incType: a ctype decorator
    #   - Examples are: c_int, c_float
    nDim = len(inMat.shape)
    
    nSize = 1
    for ii in range(0,nDim):
        nSize = nSize*inMat.shape[ii]
    
    tempMat = list(np.reshape(inMat, (nSize,),order='F'))
    
    return (incType * nSize)(*tempMat)

def funStructureMat(inList, inNNwidth, inNNdepth, inNNinput):
    outWeightMatrix = np.zeros((inNNinput+4*(2*inNNdepth-1)+inNNwidth*2*(inNNdepth-1)+1,inNNwidth))
    
    outWeightMatrix[0:inNNinput,:] = inList[0]
    outWeightMatrix[inNNinput,:] = inList[1]
    outWeightMatrix[inNNinput+1,:] = inList[2]
    outWeightMatrix[inNNinput+2,:] = inList[3]
    outWeightMatrix[inNNinput+3,:] = inList[4]
    
    for nn in range(0,inNNdepth):
        if nn == inNNdepth-1:
            STA_IDX = inNNinput+4*(2*inNNdepth-1)+inNNwidth*2*(inNNdepth-1)
            outWeightMatrix[STA_IDX:STA_IDX+1,:] = inList[-2].T
        else:
            STA_IDX = inNNinput+4+(inNNwidth*2+4*2)*nn
            STA_IDX2 = 4+1+2*(1+4)*nn
            
            outWeightMatrix[STA_IDX:(STA_IDX+inNNwidth),:] = inList[STA_IDX2]
            outWeightMatrix[STA_IDX+inNNwidth,:] = inList[STA_IDX2+1].T
            outWeightMatrix[STA_IDX+inNNwidth+1,:] = inList[STA_IDX2+2].T
            outWeightMatrix[STA_IDX+inNNwidth+2,:] = inList[STA_IDX2+3].T
            outWeightMatrix[STA_IDX+inNNwidth+3,:] = inList[STA_IDX2+4].T
            
            STA_IDX = STA_IDX + inNNwidth + 4
            
            outWeightMatrix[STA_IDX:(STA_IDX+inNNwidth),:] = inList[STA_IDX2+5]
            outWeightMatrix[STA_IDX+inNNwidth,:] = inList[STA_IDX2+6].T
            outWeightMatrix[STA_IDX+inNNwidth+1,:] = inList[STA_IDX2+7].T
            outWeightMatrix[STA_IDX+inNNwidth+2,:] = inList[STA_IDX2+8].T
            outWeightMatrix[STA_IDX+inNNwidth+3,:] = inList[STA_IDX2+9].T
    return outWeightMatrix

# -----------------------------------------------------------------------------
# PART 2: PRETRAINING VALUE FUNCTION
# -----------------------------------------------------------------------------
NSim = 1000     # Number of simulations
NIter = 50

InNN  = np.zeros((NSim,2,TT-1))      # Inputs: a(t) and d(t)
outNN = np.zeros((NSim,1,TT-1))     # Output (value function)
TD0NN = np.zeros((NSim,TT-1))

SampleCut = int(0.1*NSim)           # Validation sample (10% of whole)

PerformanceV = np.zeros((NIter+1,))
PerformanceChoice = np.zeros((TT,2*(NIter+1)))
PerformanceQ = np.zeros((TT,NIter+1))

for ss in range(0,NSim):
    tempChoice = np.random.rand(TT,2)
    # -------------------------------------------------------------------------
    # Get data based on previous result
    # -------------------------------------------------------------------------
    TempOutput = funEvalVal(tempChoice, YY, RR, [TT, bb, dd, ssigma])
    
    outNN[ss,0,:] = TempOutput[0][1:,:].T # Record: V(t+1)
    InNN[ss,:,:] = np.copy(TempOutput[1][:-1,1:].T)
    
SavePath = WorkPath+'/Models/Model0_'
TrainedList = funTrainDNN(ListVal, InNN, outNN, SampleCut, TT, SavePath)
ListVal = TrainedList[0]
Weights = TrainedList[1]

arrWeightMatrix = []
cbiaslist = []
for tt in range(0,TT-1):
    tempWeightMat = funStructureMat(Weights[tt], NNwidth, NNdepth, NNinput)
    arrWeightMatrix.append(funArraycType(tempWeightMat, c_float))
    
    cbiaslist.append(c_float(Weights[tt][-1][0]))

# -----------------------------------------------------------------------------
# Interim Evaluation
# -----------------------------------------------------------------------------
ChoiceCollectTemp = np.zeros((TT,2))

EvalstateA = np.zeros((TT-1,))
EvalstateD = np.zeros((TT-1,))

params0 = np.array([ssigma, bb*dd])
paramst = np.array([ssigma, dd])

arrparams0 = funArraycType(params0, c_float)
arrparamst = funArraycType(paramst, c_float)

GridC = np.array([41, 31, 11], dtype=int)
paddingC = np.array([1.5, 1.0, 1.0])
GridA = np.array([11, 21, 21], dtype=int)
paddingA = np.array([5.0, 1.0, 1.0])

NNstep = GridC.shape[0]

cGridC = funArraycType(GridC, c_int)
cpaddingC = funArraycType(paddingC, c_float)

cGridA = funArraycType(GridA, c_int)
cpaddingA = funArraycType(paddingA, c_float)

cNNwidth = c_int(NNwidth)
cNNdepth = c_int(NNdepth)
cNNinput = c_int(NNinput)
cNNstep = c_int(NNstep)

for tt in range(0,TT):
    if tt == 0:
        # ---------------------------------------------------------------------
        # C types conversion
        # ---------------------------------------------------------------------
        # Exogenous variables
        Exog = np.array([YY[tt], RR[tt], 0, 0])
        arrExog = funArraycType(Exog, c_float)
        
        # New choice variables
        newChoices = np.zeros((2,))
        arrChoices = funArraycType(newChoices, c_float)
        
        cVal = c_float(0)
        
        DLLChoice.FortranUpdateChoiceGrid(byref(arrWeightMatrix[tt]),
                                          cbiaslist[tt],
                                          byref(arrExog),
                                          byref(arrparams0),
                                          byref(cGridC),
                                          byref(cpaddingC),
                                          byref(cGridA),
                                          byref(cpaddingA),
                                          cNNwidth,cNNdepth,cNNinput,cNNstep,
                                          byref(arrChoices),
                                          byref(cVal))
        
        newChoices = np.array(arrChoices)
        
        ChoiceCollectTemp[tt,:] = newChoices # Update value
        
        newC = newChoices[0]
        newA = newChoices[1]
        
        CC = newC*YY[tt]
        
        EvalstateA[tt] = (YY[tt] - CC)*newA
        EvalstateD[tt] = (YY[tt] - CC)*(1-newA)

    elif tt < TT-1:
        Exog = np.array([YY[tt], RR[tt], EvalstateA[tt-1], EvalstateD[tt-1]])
        arrExog = funArraycType(Exog, c_float)
        
        # New choice variables
        newChoices = np.zeros((2,))
        arrChoices = funArraycType(newChoices, c_float)
        
        cVal = c_float(0)
        
        DLLChoice.FortranUpdateChoiceGrid(byref(arrWeightMatrix[tt]),
                                          cbiaslist[tt],
                                          byref(arrExog),
                                          byref(arrparamst),
                                          byref(cGridC),
                                          byref(cpaddingC),
                                          byref(cGridA),
                                          byref(cpaddingA),
                                          cNNwidth,cNNdepth,cNNinput,cNNstep,
                                          byref(arrChoices),
                                          byref(cVal))
        
        newChoices = np.array(arrChoices)
        
        ChoiceCollectTemp[tt,:] = newChoices # Update value
        
        newC = newChoices[0]
        newA = newChoices[1]
        
        CC = newC*(YY[tt] + EvalstateD[tt])
        
        EvalstateA[tt] = (YY[tt] - CC + RR[tt]*EvalstateA[tt-1] + EvalstateD[tt-1])*newA
        EvalstateD[tt] = (YY[tt] - CC + RR[tt]*EvalstateA[tt-1] + EvalstateD[tt-1])*(1-newA)

    else:
        ChoiceCollectTemp[tt,0] = 1
        ChoiceCollectTemp[tt,1] = 0
        
PerformanceOutput = funEvalVal(ChoiceCollectTemp, YY, RR, [TT, bb, dd, ssigma])

PerformanceV[0]          = PerformanceOutput[0][0,0]     # Record: V(t+1)
PerformanceQ[:,0]        = PerformanceOutput[1][:,0]     # Record: c(t)
PerformanceChoice[:,0:2] = ChoiceCollectTemp

pd.DataFrame(PerformanceV).to_csv(WorkPath+'/PerformanceV.csv', index=False)
pd.DataFrame(PerformanceQ).to_csv(WorkPath+'/PerformanceQ.csv', index=False)
pd.DataFrame(PerformanceChoice).to_csv(WorkPath+'/PerformanceChoice.csv', index=False)

# -----------------------------------------------------------------------------
# PART 3: MAIN TRAINING ROUTINE
# -----------------------------------------------------------------------------
epsilonmat = np.random.rand(NSim, TT)
epsilon_greedy = 0.05


for ii in range(0,NIter):
    for ss in tqdm(range(0,NSim)):
        tempChoice = np.zeros((TT,2))
        
        for tt in range(0,TT):
            if tt == 0:
                loopA = 0
                loopD = 0
                
                # By probability (1-espilon_greedy)
                if epsilonmat[ss,tt] >= epsilon_greedy:
                    # Exogenous variables
                    Exog = np.array([YY[tt], RR[tt], loopA, loopD])
                    arrExog = funArraycType(Exog, c_float)
                    
                    # New choice variables
                    newChoices = np.zeros((2,))
                    arrChoices = funArraycType(newChoices, c_float)
                    
                    cVal = c_float(0)
                    
                    DLLChoice.FortranUpdateChoiceGrid(byref(arrWeightMatrix[tt]),
                                                      cbiaslist[tt],
                                                      byref(arrExog),
                                                      byref(arrparams0),
                                                      byref(cGridC),
                                                      byref(cpaddingC),
                                                      byref(cGridA),
                                                      byref(cpaddingA),
                                                      cNNwidth,cNNdepth,cNNinput,cNNstep,
                                                      byref(arrChoices),
                                                      byref(cVal))
                    
                    tempChoice[tt,:] = np.array(arrChoices)
                    
                # By probability epsilon_greedy
                else:
                    tempChoice[tt,:] = np.random.rand(2,)
                
                CC = YY[tt]*tempChoice[tt,0]
                loopA = YY[tt]*(1-tempChoice[tt,0])*tempChoice[tt,1]
                loopD = YY[tt]*(1-tempChoice[tt,0])*(1-tempChoice[tt,1])
                
                InNN[ss,0,tt] = loopA
                InNN[ss,1,tt] = loopD
                
            elif tt < TT-1:
                if epsilonmat[ss,tt] >= epsilon_greedy:
                    Exog = np.array([YY[tt], RR[tt], loopA, loopD])
                    arrExog = funArraycType(Exog, c_float)
                    
                    # New choice variables
                    newChoices = np.zeros((2,))
                    arrChoices = funArraycType(newChoices, c_float)
                    
                    cVal = c_float(0)
                    
                    DLLChoice.FortranUpdateChoiceGrid(byref(arrWeightMatrix[tt]),
                                                      cbiaslist[tt],
                                                      byref(arrExog),
                                                      byref(arrparamst),
                                                      byref(cGridC),
                                                      byref(cpaddingC),
                                                      byref(cGridA),
                                                      byref(cpaddingA),
                                                      cNNwidth,cNNdepth,cNNinput,cNNstep,
                                                      byref(arrChoices),
                                                      byref(cVal))
                    
                    tempChoice[tt,:] = np.array(arrChoices)
                    UV = cVal.value
                    
                    CC = (YY[tt]+loopD)*tempChoice[tt,0]
                    
                    NetWorth = YY[tt]+loopD+RR[tt]*loopA
                    
                    loopA = (NetWorth-CC)*tempChoice[tt,1]
                    loopD = (NetWorth-CC)*(1-tempChoice[tt,1])
                    
                    InNN[ss,0,tt] = loopA
                    InNN[ss,1,tt] = loopD
                else:
                    tempChoice[tt,:] = np.random.rand(2,)
                    
                    CC = (YY[tt]+loopD)*tempChoice[tt,0]
                    
                    NetWorth = YY[tt]+loopD+RR[tt]*loopA
                    
                    loopA = (NetWorth-CC)*tempChoice[tt,1]
                    loopD = (NetWorth-CC)*(1-tempChoice[tt,1])
                    
                    InNN[ss,0,tt] = loopA
                    InNN[ss,1,tt] = loopD
            else:
                tempChoice[0] = 1.0
                tempChoice[1] = 0
    
    TempOutput = funEvalVal(tempChoice, YY, RR, [TT, bb, dd, ssigma])    
    outNN[ss,0,:] = TempOutput[0][1:,:].T # Record: V(t+1) 
    
    
    SavePath = WorkPath+'/Models/Model' + str(ii+1) + '_'
    TrainedList = funTrainDNN(ListVal, InNN, outNN, SampleCut, TT, SavePath)
    ListVal = TrainedList[0]
    Weights = TrainedList[1]
    
    arrWeightMatrix = []
    cbiaslist = []
    for tt in range(0,TT-1):
        tempWeightMat = funStructureMat(Weights[tt], NNwidth, NNdepth, NNinput)
        arrWeightMatrix.append(funArraycType(tempWeightMat, c_float))
        
        cbiaslist.append(c_float(Weights[tt][-1][0]))
    
    # -------------------------------------------------------------------------
    # Interim Evaluation
    # -------------------------------------------------------------------------
    EvalstateA = np.zeros((TT-1,))
    EvalstateD = np.zeros((TT-1,))

    for tt in range(0,TT):
        if tt == 0:
            # ---------------------------------------------------------------------
            # C types conversion
            # ---------------------------------------------------------------------
            # Exogenous variables
            Exog = np.array([YY[tt], RR[tt], 0, 0])
            arrExog = funArraycType(Exog, c_float)
            
            # New choice variables
            newChoices = np.zeros((2,))
            arrChoices = funArraycType(newChoices, c_float)
            
            cVal = c_float(0)
            
            DLLChoice.FortranUpdateChoiceGrid(byref(arrWeightMatrix[tt]),
                                              cbiaslist[tt],
                                              byref(arrExog),
                                              byref(arrparams0),
                                              byref(cGridC),
                                              byref(cpaddingC),
                                              byref(cGridA),
                                              byref(cpaddingA),
                                              cNNwidth,cNNdepth,cNNinput,cNNstep,
                                              byref(arrChoices),
                                              byref(cVal))
            
            newChoices = np.array(arrChoices)
            
            ChoiceCollectTemp[tt,:] = newChoices # Update value
            
            newC = newChoices[0]
            newA = newChoices[1]
            
            CC = newC*YY[tt]
            
            EvalstateA[tt] = (YY[tt] - CC)*newA
            EvalstateD[tt] = (YY[tt] - CC)*(1-newA)
    
        elif tt < TT-1:
            Exog = np.array([YY[tt], RR[tt], EvalstateA[tt-1], EvalstateD[tt-1]])
            arrExog = funArraycType(Exog, c_float)
            
            # New choice variables
            newChoices = np.zeros((2,))
            arrChoices = funArraycType(newChoices, c_float)
            
            cVal = c_float(0)
            
            DLLChoice.FortranUpdateChoiceGrid(byref(arrWeightMatrix[tt]),
                                              cbiaslist[tt],
                                              byref(arrExog),
                                              byref(arrparamst),
                                              byref(cGridC),
                                              byref(cpaddingC),
                                              byref(cGridA),
                                              byref(cpaddingA),
                                              cNNwidth,cNNdepth,cNNinput,cNNstep,
                                              byref(arrChoices),
                                              byref(cVal))
            
            newChoices = np.array(arrChoices)
            
            ChoiceCollectTemp[tt,:] = newChoices # Update value
            
            newC = newChoices[0]
            newA = newChoices[1]
            
            CC = newC*(YY[tt] + EvalstateD[tt])
            
            EvalstateA[tt] = (YY[tt] - CC + RR[tt]*EvalstateA[tt-1] + EvalstateD[tt-1])*newA
            EvalstateD[tt] = (YY[tt] - CC + RR[tt]*EvalstateA[tt-1] + EvalstateD[tt-1])*(1-newA)
    
        else:
            ChoiceCollectTemp[tt,0] = 1
            ChoiceCollectTemp[tt,1] = 0
        
    PerformanceOutput = funEvalVal(ChoiceCollectTemp, YY, RR, [TT, bb, dd, ssigma])
            
    PerformanceV[ii+1]          = PerformanceOutput[0][0,0]     # Record: V(t+1)
    PerformanceQ[:,ii+1]        = PerformanceOutput[1][:,0]     # Record: c(t)
    PerformanceChoice[:,(ii+1)*2:(ii+2)*2] = ChoiceCollectTemp
    
    pd.DataFrame(PerformanceV).to_csv(WorkPath+'/PerformanceV.csv', index=False)
    pd.DataFrame(PerformanceQ).to_csv(WorkPath+'/PerformanceQ.csv', index=False)
    pd.DataFrame(PerformanceChoice).to_csv(WorkPath+'/PerformanceChoice.csv', index=False)