import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
os.chdir('../')
import hapi
os.environ["GAAS_OCL_DEVICE"] = '0'
import gaas_ocl as gs
import numpy as np
import time
import random
import pandas as pd
import os
import matplotlib.pyplot as plt

def getError(nus_h,coefs_h,nus_g,coefs_g): #gets percent error
    err = 0
    sum = 0
    length = min(len(coefs_g),len(coefs_h))
    for i in range(length):
        err+=abs(coefs_h[i]-coefs_g[i])
        sum+=coefs_h[i]
    return err/sum*100 #%

def hapiSimVoigt_raw(features, wavenumStep, startWavenum, endWavenum):
    """
    Runs HTP simulation using HAPI
    :param features: list of gaas.VoigtRawStructDatatype objects
    :param wavenumStep: wavenumber resolution
    :param startWavenum: wavenumber range start
    :param endWavenum: wavenumber range end
    :return: ( wavenums, spectrum)
    """

    def toWavenumIndex(v):
        return int((v-startWavenum)/wavenumStep)

    totWavenumCount = (endWavenum-startWavenum)/wavenumStep

    wavenums = np.arange(startWavenum,endWavenum,wavenumStep)
    spectrum = np.zeros_like(wavenums)
    for feat in features:
        (ia,lc,GamD,Gam0) = feat.getDataTuple()
        maxHW = max(GamD,Gam0)
        minWvn = lc-maxHW*50
        maxWvn = lc+maxHW*50
        minInd = toWavenumIndex(minWvn)+1
        maxInd = toWavenumIndex(maxWvn)+1
        n = maxInd-minInd
        # print(feat)
        indStart = max(0,min(minInd,wavenums.size))
        indEnd = max(0,min(maxInd,wavenums.size))
        lineshapeVals = hapi.PROFILE_VOIGT(lc,GamD,Gam0,wavenums[indStart:indEnd])[0]
        spectrum[indStart:indEnd]+=ia*lineshapeVals
        # for i in range(n):
        #     ind = minInd+i
        #     if(ind >= wavenums.size):
        #         continue
        #     wvn = wavenums[ind]
        #     val = ia*hapi.PROFILE_VOIGT(lc,GamD,Gam0,wvn)[0]
        #     spectrum[ind]+=val
    return wavenums,spectrum

def getErrorVoigt_raw(features : list[gs.VoigtRawFeatureData], wavenumStep : float , startWavenum: float, endWavenum : float) :
    print("Priming gaas")
    (wvn_gs,abs_gs) = gs.simVoigtRaw(features,wavenumStep,startWavenum,endWavenum)

    print("Running GAAS")

    t1 = time.time()
    (wvn_gs,abs_gs) = gs.simVoigtRaw(features,wavenumStep,startWavenum,endWavenum)
    gTime = time.time() - t1

    print("Running HAPI")
    t1 = time.time()
    nus_h,coefs_h = hapiSimVoigt_raw(features, wavenumStep, startWavenum, endWavenum)
    hTime = time.time() - t1
    err = getError(nus_h,coefs_h,wvn_gs,abs_gs)
    print("GAASTime: ",gTime)
    print("HAPITime: ",hTime)
    # plt.plot(nus_h, coefs_h)
    # plt.plot(wvn_gs, abs_gs)
    # plt.plot(wvn_gs, coefs_h-abs_gs)
    # plt.show()
    return err, hTime, gTime
    
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
    out = []

    t1 = time.time()
    (wvn_gs,abs_gs) = gs.simVoigtRaw(featureSets[0],wvnStep,startWvn,endWvn)
    gTime = time.time() - t1

    t1 = time.time()
    nus_h,coefs_h = hapiSimVoigt_raw(featureSets[0],wvnStep,startWvn,endWvn)
    hTime = time.time() - t1
    plt.plot(nus_h, coefs_h)
    plt.plot(wvn_gs, abs_gs)
    plt.show()
    
    outPD = pd.DataFrame(np.stack([nus_h,coefs_h,wvn_gs,abs_gs], axis=1),columns=["wvn_hapi","coefs_hapi","wvn_gaas","coefs_gaas"])
    return outPD

class randomFeatGenerator:
    def __init__(self, seed : int) -> None:
        random.seed(seed)
        self.min_val = 0.08
        self.max_val = 0.12

    def genFeatures(self, startWavenum : float, endWavenum : float, nFeatures: int) -> list[gs.VoigtRawFeatureData]:
        feat_data = []
        for i in range(nFeatures):
            lc = random.uniform(startWavenum,endWavenum)
            GamD = random.uniform(self.min_val,self.max_val)
            Gam0 = random.uniform(self.min_val,self.max_val)
            lineIntensity = random.uniform(self.min_val,self.max_val)
            feat_data.append(gs.VoigtRawFeatureData(lc,lineIntensity,GamD,Gam0))
        return feat_data
    
    def genNonRandomFeatures(self, startWavenum : float, endWavenum : float, nFeatures: int, const_GamD:float=None, const_Gam0:float=None) -> list[gs.VoigtRawFeatureData]:
        #generates a list of evenly spaced features of the same size
        feat_data = []
        d_lc = (endWavenum-startWavenum)/nFeatures
        for i in range(nFeatures):
            lc = startWavenum + d_lc*i
            if(const_GamD != None):
                GamD = const_GamD
            else:
                GamD = (self.max_val-self.min_val)/2
            if(const_Gam0 != None):
                Gam0 = const_Gam0
            else:    
                Gam0 = (self.max_val-self.min_val)/2
            lineIntensity = (self.max_val-self.min_val)/2
            feat_data.append(gs.VoigtRawFeatureData(lc,lineIntensity,GamD,Gam0))
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
        err, hTime, gTime = getErrorVoigt_raw(featureSets[i], wvnStep, startWvn, endWvn)
        out.append([err,hTime,gTime])
        # print("error: ",err)
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
            feats.append(featGen.genNonRandomFeatures(startWvn,endWvn,int(n)))
        else:
            feats.append(featGen.genFeatures(startWvn,endWvn,int(n)))

    out = []
    for i in range(len(feats)):
        err, hTime, gTime = getErrorVoigt_raw(feats[i], wvnStep, startWvn, endWvn)
        out.append([len(feats[i]),err,hTime,gTime])

    outpd = pd.DataFrame(out, columns=["numFeats","error %","HAPITime","gaasTime"])
    return outpd

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
            feats = featGen.genNonRandomFeatures(startWvn,endWvn,int(i_nFeats),j_hwhm/2,j_hwhm/2)
            err, hTime, gTime = getErrorVoigt_raw(feats, wvnStep, startWvn, endWvn)
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
            feats = featGen.genNonRandomFeatures(startWvn,endWvn,int(i_nFeats),j_hwhm/2,j_hwhm/2)
            t1 = time.time()
            # (wvn_gs,abs_gs) = gs.simHTP(feats[i],300,1.0,wvnStep,startWvn,endWvn)
            (wvn_gs,abs_gs) = gs.simVoigtRaw(feats,wvnStep,startWvn,endWvn)

            # plt.plot(wvn_gs,abs_gs)
            # plt.show()
            gTime = time.time() - t1
            print("time: ",gTime)
            out.append([int(i_nFeats),j_hwhm,gTime])

    outpd = pd.DataFrame(out, columns=["numFeats", "meanHWHM", "gaasTime"])
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
            print("Running GAAS Voigt")
            print("nfeats: ",len(feats[i]))
            #prime it by running once before
            t1 = time.time()
            # (wvn_gs,abs_gs) = gs.simHTP(feats[i],300,1.0,wvnStep,startWvn,endWvn)
            (wvn_gs,abs_gs) = gs.simVoigtRaw(feats[i],wvnStep,startWvn,endWvn)

            # plt.plot(wvn_gs,abs_gs)
            # plt.show()
            gTime = time.time() - t1
            print("time: ",gTime)

            out.append([i_run, len(feats[i]),gTime])

    outpd = pd.DataFrame(out, columns=["run_num","numFeats","gaasTime"])
    return outpd
    
def runAll():
    cwd = os.path.dirname(os.path.realpath(__file__))
    # compPlotDF = genComparisonPlot_data()
    # compPlotDF.to_csv(cwd+"\\voigt_comparison_plot.csv")
    
    randValRes = runRandValidation(25600, 20)
    randValRes.to_csv(cwd+"\\voigt_rand_val_new_adaptive.csv")
    # speedValRes = runSpeedTest(1024000, 20, 1)
    # speedValRes.to_csv(cwd+"//voigt_speed_val_new_adaptive.csv")
    # speedGAASRes = runSpeedTestGaasOnly(1024000, 20, 1)
    # speedGAASRes.to_csv(cwd+"\\voigt_speed_val_new_adaptive_gaas.csv")

    # res = runSpeedTest2d(10000, 0.1, 10, 10, 1)
    # res = runSpeedTest2d_gaasonly(10000, 0.1, 10, 10, 1)
    # res.to_csv(cwd+"\\voigt_speed_test_2d_gaas_new_adaptive.csv")
if(__name__ == "__main__"):
    runAll()