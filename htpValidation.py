import hapi
import gaas_ocl as gs
import numpy as np
import time
import random
import pandas as pd
import os

def getError(nus_h,coefs_h,nus_g,coefs_g): #gets percent error
    err = 0
    sum = 0
    length = min(len(coefs_g),len(coefs_h))
    for i in range(length):
        err+=abs(coefs_h[i]-coefs_g[i])
        sum+=coefs_h[i]
    return err/sum*100 #%

def hapiSimHTP(features, tempK, molarMass, wavenumStep, startWavenum, endWavenum):
    """
    Runs HTP simulation using HAPI
    :param features: list of gaas.HTPFeatureData objects
    :param tempK: temperature in Kelvin
    :param molarMass: molar mass of absorber
    :param wavenumStep: wavenumber resolution
    :param startWavenum: wavenumber range start
    :param endWavenum: wavenumber range end
    :return: ( wavenums, spectrum)
    """

    def toWavenumIndex(v):
        return int((v-startWavenum)/wavenumStep)

    def dopplerHWHM(transWavenum, molarMass, tempK):
        LOG2 = 0.69314718056

        cMassMol = 1.66053873e-27
        cBolts = 1.380648813E-16 # erg/K
        cc = 2.99792458e10;		 # cm/s
        m = molarMass * cMassMol * 1000
        return np.sqrt(2 * cBolts * tempK * np.log(2) / m / (cc * cc)) * transWavenum

    # this is designed to be as similar as gaas::HTPLineshape::lineshapeHTP in Gaas.cu as possible
    totWavenumCount = (endWavenum-startWavenum)/wavenumStep

    wavenums = np.arange(startWavenum,endWavenum,wavenumStep)
    spectrum = np.zeros_like(wavenums)

    for feat in features:
        (linecenter,Gam0,Gam2,Delta0,Delta2,anuVC,eta,lineIntensity) = feat.getDataTuple()
        gammaD = dopplerHWHM(linecenter,molarMass,tempK)
        maxHW = max(gammaD,Gam0,Gam2)
        minWvn = linecenter-maxHW*50
        maxWvn = linecenter+maxHW*50
        minInd = toWavenumIndex(minWvn)
        maxInd = toWavenumIndex(maxWvn)
        n = maxInd-minInd
        # print(feat)
        for i in range(n):
            ind = minInd+i
            if(ind >= wavenums.size):
                continue
            wvn = wavenums[ind]
            val = lineIntensity*hapi.PROFILE_HT(linecenter,gammaD,Gam0,Gam2,Delta0,Delta2,anuVC,eta,wvn,Ylm=0.0)[0]
            spectrum[ind]+=val
    return wavenums,spectrum

def getErrorHTP(features : list[gs.HTPFeatureData], tempK : float, molarMass : float, wavenumStep : float , startWavenum: float, endWavenum : float) :

    print("Running GAAS HTP")
    t1 = time.time()
    (wvn_gs,abs_gs) = gs.simHTP_legacy(features,tempK,molarMass,wavenumStep,startWavenum,endWavenum)
    gTime = time.time() - t1

    print("Running HAPI")
    t1 = time.time()
    nus_h,coefs_h = hapiSimHTP(features, tempK, molarMass, wavenumStep, startWavenum, endWavenum)
    hTime = time.time() - t1

    err = getError(nus_h,coefs_h,wvn_gs,abs_gs)
    return err, hTime, gTime
    
class randomFeatGenerator:
    def __init__(self, seed : int) -> None:
        random.seed(seed)
        self.min_val = 0.001
        self.max_val = 0.1

    def genFeatures(self, startWavenum : float, endWavenum : float, nFeatures: int) -> list[gs.HTPFeatureData]:
        feat_data = []
        for i in range(nFeatures):
            lc = random.uniform(startWavenum,endWavenum)
            Gam0 = random.uniform(self.min_val,self.max_val)
            Gam2 = random.uniform(self.min_val,self.max_val)
            Delta0 = random.uniform(self.min_val,self.max_val)
            Delta2 = random.uniform(self.min_val,self.max_val)
            anuVC = random.uniform(self.min_val,self.max_val)
            eta = random.uniform(self.min_val,self.max_val)
            lineIntensity = random.uniform(self.min_val,self.max_val)
            feat_data.append(gs.HTPFeatureData(lc,Gam0,Gam2,Delta0,Delta2,anuVC,eta,lineIntensity))
        return feat_data
    
def runRandValidation(numFeatures: int, numRuns: int) -> pd.DataFrame:
    featureSets = []
    startWvn = 2000
    endWvn = 6000
    wvnStep = 0.001
    seed = 1
    featGen = randomFeatGenerator(seed)
    #gen features
    for i in range(numRuns):
        featureSets.append(featGen.genFeatures(startWvn,endWvn,numFeatures))

    out = []
    for i in range(numRuns):
        err, hTime, gTime = getErrorHTP(featureSets[i], 300, 1.0, wvnStep, startWvn, endWvn)
        out.append([err,hTime,gTime])
    outpd = pd.DataFrame(out, columns=["error %","HAPITime","gaasTime"])

    return outpd
    
def runSpeedTest(maxFeats: int, numRuns: int, randSeed:int):
    startWvn = 2000
    endWvn = 6000
    wvnStep = 0.001
    featGen = randomFeatGenerator(randSeed)
    nfeats = np.arange(1,maxFeats,maxFeats/numRuns)
    feats = []
    for n in nfeats:
        feats.append(featGen.genFeatures(startWvn,endWvn,n))

    out = []
    for i in range(feats):
        err, hTime, gTime = getErrorHTP(feats[i], 300, 1.0, wvnStep, startWvn, endWvn)
        out.append([len(feats[i]),err,hTime,gTime])

    outpd = pd.DataFrame(out, columns=["numFeats","error %","HAPITime","gaasTime"])
    return outpd
    
def runAll():
    cwd = os.path.dirname(os.path.realpath(__file__))
    randValRes = runRandValidation(512, 20)
    randValRes.to_csv(cwd+"\\Validation\\htp_rand_val.csv")
    speedValRes = runSpeedTest(512, 20, 1)
    speedValRes.to_csv(cwd+"\\Validation\\htp_speed_val.csv")
