import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
os.chdir('../')

import hapi
import gaas_ocl as gs
import numpy as np
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

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
    # iter=0
    for feat in features:
        # print(iter)
        (linecenter,Gam0,Gam2,Delta0,Delta2,anuVC,eta,lineIntensity) = feat.getDataTuple()
        gammaD = dopplerHWHM(linecenter,molarMass,tempK)
        maxHW = max(gammaD,Gam0,Gam2)
        minWvn = linecenter-maxHW*50
        maxWvn = linecenter+maxHW*50
        minInd = toWavenumIndex(minWvn)
        maxInd = toWavenumIndex(maxWvn)
        n = maxInd-minInd
        # print(feat)
        indStart = max(0,min(minInd,wavenums.size))
        indEnd = max(0,min(maxInd,wavenums.size))
        lineshapeVals = hapi.PROFILE_HT(linecenter,gammaD,Gam0,Gam2,Delta0,Delta2,anuVC,eta,wavenums[indStart:indEnd],Ylm=0.0)[0]
        spectrum[indStart:indEnd]+=lineIntensity*lineshapeVals

        # for i in range(n):
        #     ind = minInd+i
        #     if(ind >= wavenums.size):
        #         continue
        #     wvn = wavenums[ind]
        #     val = lineIntensity*hapi.PROFILE_HT(linecenter,gammaD,Gam0,Gam2,Delta0,Delta2,anuVC,eta,wvn,Ylm=0.0)[0]
        #     spectrum[ind]+=val
        # iter+=1
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
    # plt.plot(nus_h, coefs_h)
    # plt.plot(wvn_gs, abs_gs)
    # plt.show()
    print("err: ",err, "hTime: ", hTime, "gTime: ",gTime)
    return err, hTime, gTime
    
class randomFeatGenerator:
    def __init__(self, seed : int) -> None:
        random.seed(seed)
        self.min_val = 0.08
        self.max_val = 0.12

    def genFeatures(self, startWavenum : float, endWavenum : float, nFeatures: int) -> list[gs.HTPFeatureData]:
        feat_data = []
        for i in range(nFeatures):
            lc = random.uniform(startWavenum,endWavenum)
            Gam2 = random.uniform(self.min_val,self.max_val)

            Gam0 = random.uniform(self.min_val,self.max_val) + Gam2
            Delta2 = random.uniform(self.min_val,self.max_val)
            Delta0 = random.uniform(self.min_val,self.max_val) + Delta2
            anuVC = random.uniform(self.min_val,self.max_val)
            eta = random.uniform(self.min_val,self.max_val)
            lineIntensity = random.uniform(self.min_val,self.max_val)
            feat_data.append(gs.HTPFeatureData(lc,Gam0,Gam2,Delta0,Delta2,anuVC,eta,lineIntensity))
        return feat_data
    
    def genNonRandomFeatures(self, startWavenum : float, endWavenum : float, nFeatures: int) -> list[gs.HTPFeatureData]:
        #generates a list of evenly spaced features of the same size
        feat_data = []
        d_lc = (endWavenum-startWavenum)/nFeatures
        for i in range(nFeatures):
            lc = startWavenum + d_lc*i
            Gam0 = (self.max_val-self.min_val)/2
            Gam2 = (self.max_val-self.min_val)/2
            Delta0 = (self.max_val-self.min_val)/2
            Delta2 = (self.max_val-self.min_val)/2
            anuVC = (self.max_val-self.min_val)/2
            eta = (self.max_val-self.min_val)/2
            lineIntensity = (self.max_val-self.min_val)/2
            feat_data.append(gs.HTPFeatureData(lc,Gam0,Gam2,Delta0,Delta2,anuVC,eta,lineIntensity))
        return feat_data
    
    def genNonRandomFeatures(self, startWavenum : float, endWavenum : float, nFeatures: int, hwhm: float) -> list[gs.HTPFeatureData]:
        #generates a list of evenly spaced features of the same size
        feat_data = []
        d_lc = (endWavenum-startWavenum)/nFeatures
        for i in range(nFeatures):
            lc = startWavenum + d_lc*i
            Gam0 = hwhm
            Gam2 = 0.0001
            Delta0 = (self.max_val-self.min_val)/2
            Delta2 = 0.0001
            anuVC = (self.max_val-self.min_val)/2
            eta = (self.max_val-self.min_val)/2
            lineIntensity = (self.max_val-self.min_val)/2
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
        out.append([err/100,hTime,gTime])
    outpd = pd.DataFrame(out, columns=["error %","HAPITime","gaasTime"])

    return outpd
    
def runSpeedTest(maxFeats: int, numRuns: int, randSeed:int) -> pd.DataFrame:
    startWvn = 2000
    endWvn = 6000
    wvnStep = 0.001
    evenSpacedFeats = True
    featGen = randomFeatGenerator(randSeed)
    nfeats = np.arange(1,maxFeats,maxFeats/numRuns)
    feats = []
    delta_lc = (endWvn-startWvn)/numRuns
    for n in nfeats:
        if evenSpacedFeats:
            feats.append(featGen.genNonRandomFeatures(startWvn,endWvn,int(n),0.2))
        else:
            feats.append(featGen.genFeatures(startWvn,endWvn,int(n)))

    out = []
    for i in range(len(feats)):
        err, hTime, gTime = getErrorHTP(feats[i], 300, 1.0, wvnStep, startWvn, endWvn)
        out.append([len(feats[i]),err,hTime,gTime])

    outpd = pd.DataFrame(out, columns=["numFeats","error %","HAPITime","gaasTime"])
    return outpd
    
def runSpeedTestGaasOnly(maxFeats: int, numRuns: int, randSeed:int) -> pd.DataFrame:
    startWvn = 2000
    endWvn = 20000
    wvnStep = 0.001
    evenSpacedFeats = True
    featGen = randomFeatGenerator(randSeed)
    nfeats = np.arange(1,maxFeats,maxFeats/numRuns)
    feats = []
    nRuns = 10
    delta_lc = (endWvn-startWvn)/numRuns
    for n in nfeats:
        if evenSpacedFeats:
            feats.append(featGen.genNonRandomFeatures(startWvn,endWvn,int(n)))
        else:
            feats.append(featGen.genFeatures(startWvn,endWvn,int(n)))

    out = []
    for i_run in range(nRuns):
        for i in range(len(feats)):
            print("Running GAAS HTP")
            print("nfeats: ",len(feats[i]))
            #prime it by running once before
            t1 = time.time()
            (wvn_gs,abs_gs) = gs.simHTP_legacy(feats[i],300,1.0,wvnStep,startWvn,endWvn)
            # plt.plot(wvn_gs,abs_gs)
            # plt.show()
            gTime = time.time() - t1
            print("time: ",gTime)

            out.append([i_run, len(feats[i]),gTime])

    outpd = pd.DataFrame(out, columns=["run_num","numFeats","gaasTime"])
    return outpd
    
def runSingleGaasOnly(maxFeats: int, numRuns: int, randSeed:int) -> pd.DataFrame:
    startWvn = 2000
    endWvn = 20000
    wvnStep = 0.001
    evenSpacedFeats = False
    featGen = randomFeatGenerator(randSeed)
    nfeats = [200534]#np.arange(1,maxFeats,maxFeats/numRuns)
    feats = []
    nRuns = 10
    delta_lc = (endWvn-startWvn)/numRuns
    for n in nfeats:
        if evenSpacedFeats:
            feats.append(featGen.genNonRandomFeatures(startWvn,endWvn,int(n)))
        else:
            feats.append(featGen.genFeatures(startWvn,endWvn,int(n)))

    out = []
    for i_run in range(nRuns):
        
        for i in range(len(feats)):
            print("Running GAAS HTP")
            print("nfeats: ",len(feats[i]))
            #prime it by running once before
            t1 = time.time()
            (wvn_gs,abs_gs) = gs.simHTP_legacy(feats[i],300,1.0,wvnStep,startWvn,endWvn)

            plt.plot(wvn_gs,abs_gs)
            plt.show()
            gTime = time.time() - t1
            print("time: ",gTime)

            out.append([i_run, len(feats[i]),gTime])

        if(evenSpacedFeats == False):
            feats = []
            for n in nfeats:
                feats.append(featGen.genFeatures(startWvn,endWvn,int(n)))

    outpd = pd.DataFrame(out, columns=["run_num","numFeats","gaasTime"])
    return outpd

def genComparisonPlot_data()->pd.DataFrame:
    featureSets = []
    startWvn = 2000
    endWvn = 6000
    wvnStep = 0.001
    seed = 1
    featGen = randomFeatGenerator(seed)
    numFeatures = 1280
    #gen features
    featureSets.append(featGen.genFeatures(startWvn,endWvn,numFeatures))

    t1 = time.time()
    (wvn_gs,abs_gs) = gs.simHTP_legacy(featureSets[0],300,1, wvnStep,startWvn,endWvn)
    gTime = time.time() - t1

    print("Running HAPI")
    t1 = time.time()
    nus_h,coefs_h = hapiSimHTP(featureSets[0], 300, 1, wvnStep,startWvn,endWvn)
    print(np.sum(np.abs(coefs_h-abs_gs)) / np.sum(coefs_h))
    hTime = time.time() - t1
    plt.plot(nus_h,coefs_h)
    plt.plot(wvn_gs,abs_gs)
    plt.show()
    outPD = pd.DataFrame(np.stack([nus_h,coefs_h,wvn_gs,abs_gs], axis=1),columns=["wvn_hapi","coefs_hapi","wvn_gaas","coefs_gaas"])
    return outPD

def runSpeedTest2d(maxFeats: int, minHWHM: int, maxHWHM: int, numRuns: int, randSeed:int) -> pd.DataFrame:
    startWvn = 2000
    endWvn = 6000
    wvnStep = 0.001
    evenSpacedFeats = True
    featGen = randomFeatGenerator(randSeed)

    nfeats = np.arange(1,maxFeats,maxFeats/numRuns)
    hwhms = np.linspace(minHWHM,maxHWHM,numRuns)

    feats = []
    out = []

    for j in range(numRuns): #hwhm
        for i in range(numRuns): #nlines
            i_nFeats = nfeats[i]
            j_hwhm = hwhms[j]
            feats = featGen.genNonRandomFeatures(startWvn,endWvn,int(i_nFeats),j_hwhm)
            print("nlines: ",i_nFeats)
            err, hTime, gTime = getErrorHTP(feats, 300, 1.0, wvnStep, startWvn, endWvn)
            out.append([int(i_nFeats),j_hwhm,err,hTime,gTime])

    outpd = pd.DataFrame(out, columns=["numFeats", "meanHWHM", "error %", "HAPITime", "gaasTime"])
    return outpd

def runSpeedTest2d_gaasonly(maxFeats: int, minHWHM: int, maxHWHM: int, numRuns: int, randSeed:int) -> pd.DataFrame:
    startWvn = 2000
    endWvn = 6000
    wvnStep = 0.001
    evenSpacedFeats = True
    featGen = randomFeatGenerator(randSeed)

    nfeats = np.arange(1,maxFeats,maxFeats/numRuns)
    hwhms = np.linspace(minHWHM,maxHWHM,numRuns)

    feats = []
    out = []

    for j in range(numRuns): #hwhm
        for i in range(numRuns): #nlines
            i_nFeats = nfeats[i]
            j_hwhm = hwhms[j]
            feats = featGen.genNonRandomFeatures(startWvn,endWvn,int(i_nFeats),j_hwhm)
            t1 = time.time()
            (wvn_gs,abs_gs) = gs.simHTP_legacy(feats,300,1.0,wvnStep,startWvn,endWvn)
            # (wvn_gs,abs_gs) = gs.simVoigtRaw(feats,wvnStep,startWvn,endWvn)

            # plt.plot(wvn_gs,abs_gs)
            # plt.show()
            gTime = time.time() - t1
            print("time: ",gTime)
            out.append([int(i_nFeats),j_hwhm,gTime])

    outpd = pd.DataFrame(out, columns=["numFeats", "meanHWHM", "gaasTime"])
    return outpd

def runAll():
    cwd = os.path.dirname(os.path.realpath(__file__))
    # compPlotDF = genComparisonPlot_data()
    # compPlotDF.to_csv(cwd+"/gaas_with_approx/htp_comparison_plot.csv")
    # randValRes = runRandValidation(25600, 20)
    # randValRes.to_csv(cwd+"/gaas_with_approx/htp_rand_val.csv")
    # speedValRes = runSpeedTest(25600, 20, 1)
    # speedValRes.to_csv(cwd+"/gaas_with_approx/htp_speed_val.csv")
    s2dRes = runSpeedTest2d(10000, 0.1, 10, 10, 1)
    s2dRes.to_csv(cwd+"/gaas_with_approx/htp_2d.csv")

    # speedGAASRes = runSpeedTestGaasOnly(256000, 60, 1)
    # HT2d = runSpeedTest2d_gaasonly(10000, 0.1, 10, 10, 1)
    # HT2d.to_csv(cwd+"\\htp_2d.csv")
    # runSingleGaasOnly(256000, 60, 1)
    # speedGAASRes.to_csv(cwd+"\\htp_speed_gaas_random.csv")
runAll()