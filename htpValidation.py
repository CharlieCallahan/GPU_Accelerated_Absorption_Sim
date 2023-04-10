import hapi
import gaas
import numpy as np

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