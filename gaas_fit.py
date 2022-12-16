import gaas as gs
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import htpValidation as htpVal
import random
from lmfit import Parameters, minimize, fit_report

nFeatures = 128
randSeed = 1

startWavenum = 7000
endWavenum = 20000
wavenumStep = 0.05 #wavenums per simulation step
molarMass = 1.0
tempK = 300

feat_wavenums = np.linspace(startWavenum,endWavenum,nFeatures)
feat_data = []
#generate random lines
for lc in feat_wavenums:
    dlc    =    0.0
    Gam0   =   0.1 + abs(random.random())
    Gam2   =   abs(random.random())*Gam0*0.5
    Delta0 = 0.1 + abs(random.random())
    Delta2 = abs(random.random())*Delta0*0.5 #abs(random.random())*Gam0*0.5
    anuVc  =  abs(random.random()) 
    eta    =    abs(random.random())
    lineIntensity = abs(random.random())*0.1
    feat_data.append(gs.HTPFeatureData(lc+dlc,Gam0,Gam2,Delta0,Delta2,anuVc,eta,lineIntensity))

t0 = time.time()
wvn_sim, spec_sim = gs.simHTP(feat_data,tempK,molarMass,wavenumStep,startWavenum,endWavenum)
gaasTime = time.time()-t0
print("gaas time: ",gaasTime)

plt.plot(wvn_sim,spec_sim)
plt.show()

#converts between lm fit params and htpFeatureData struct list
def pars_to_feat_data(pars : Parameters):
    ind = 0
    feat_list = []
    vdict = pars.valuesdict()
    for key in vdict:
        featIndex = int(key[1:8])
        tupleInd = int(key[8])
        if(featIndex>len(feat_list)-1):
            #add to feat_list
            while(len(feat_list)-1 < featIndex):
                feat_list.append(gs.HTPFeatureData(0,0,0,0,0,0,0,0))
        
        feat_list[featIndex].dataList[tupleInd] = vdict[key]
        # print(feat_list[featIndex].dataList)
    return feat_list

#converts feature data to lmfit pars
def feat_data_to_pars(feat_data_guess):
    out = Parameters()
    index = 0
    for fd in feat_data_guess:
        (linecenter,Gam0,Gam2,Delta0,Delta2,anuVC,eta,lineIntensity) = fd.getDataTuple()

        out.add("n"+f'{index:07d}' + "0",value = linecenter,   vary=False)
        out.add("n"+f'{index:07d}' + "1",value = Gam0      ,   vary=True)
        out.add("n"+f'{index:07d}' + "2",value = Gam2      ,   vary=True)
        out.add("n"+f'{index:07d}' + "3",value = Delta0    ,   vary=True)
        out.add("n"+f'{index:07d}' + "4",value = Delta2    ,   vary=True)
        out.add("n"+f'{index:07d}' + "5",value = anuVC     ,   vary=True)
        out.add("n"+f'{index:07d}' + "6",value = eta       ,   vary=True)
        out.add("n"+f'{index:07d}' + "7",value = lineIntensity,vary=True)
        index+=1
    return out
resMags = []

def _residual(pars, data=None, eps=None):
    fd = pars_to_feat_data(pars)
    wvn,spec = gs.simHTP(fd,tempK,molarMass,wavenumStep,startWavenum,endWavenum)
    res = np.nan_to_num(np.array(spec)) - np.array(spec_sim)
    resSum = np.sum(np.abs(res))
    print("\t\t\t\t\t\t\t",resSum)
    resMags.append(resSum)

    if(len(resMags) % 1000 == 0):
        plt.plot(resMags)
        plt.show()
    
    return res

feat_guess = []
for lc in feat_wavenums:
    dlc    =    0.0
    Gam0   =   0.1 + abs(random.random())
    Gam2   =   abs(random.random())*Gam0*0.5
    Delta0 = 0.1 + abs(random.random())
    Delta2 = abs(random.random())*Delta0*0.5 #abs(random.random())*Gam0*0.5
    anuVc  =  abs(random.random()) 
    eta    =    abs(random.random())
    lineIntensity = abs(random.random())*0.1
    feat_guess.append(gs.HTPFeatureData(lc+dlc,Gam0,Gam2,Delta0,Delta2,anuVc,eta,lineIntensity))

# print("NUM FEATS INIT: ",len(feat_guess))
# print("TEST: ",len(pars_to_feat_data(feat_data_to_pars(feat_guess))))
# exit()
guess_pars = feat_data_to_pars(feat_guess)
result = minimize(_residual,guess_pars,epsfcn=0.001)
print("done fitting\nResults:")
print(fit_report(result))
# t = feat_data_to_pars(feat_data)
# pars_to_feat_data(t)

#fitting funtions
