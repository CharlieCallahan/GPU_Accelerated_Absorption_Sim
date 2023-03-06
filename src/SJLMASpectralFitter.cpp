#include "SJLMASpectralFitter.hpp"
#include "Gaas.cuh"

SJLMSpectrumFitTask::SJLMSpectrumFitTask(float* absSpectrum, 
                    double startWavenum,
                    double endWavenum,
                    double wavenumStep,
                    int spec_len,
                    int nFeatures,
                    float tempK,
                    float molarMass,
                    featureDataHTP* guesses) : LMATask(1.0)
{
    this->specVec = new CudaLMVec(absSpectrum,spec_len);
    float* tempGuessVec = new float [nFeatures*8];

    for (int i = 0; i < nFeatures; i++){
        tempGuessVec[i*8+0] = float(guesses[i].linecenter);
        tempGuessVec[i*8+1] = guesses[i].Gam0; //Gamma0
        tempGuessVec[i*8+2] = guesses[i].Gam2; //Gamma2
        tempGuessVec[i*8+3] = guesses[i].Delta0; //shift0
        tempGuessVec[i*8+4] = guesses[i].Delta2; //shift2
        tempGuessVec[i*8+5] = guesses[i].anuVC; //nuVC
        tempGuessVec[i*8+6] = guesses[i].eta; //eta
        tempGuessVec[i*8+7] = guesses[i].lineIntensity; //line intensity
    }

    this->guessVec = new CudaLMVec(tempGuessVec,nFeatures*8);
    this->nFeatures = nFeatures;
    this->startWavenum = startWavenum;
    this->endWavenum = endWavenum;
    this->wavenumStep = wavenumStep;
    this->spec_len = spec_len;

    delete[] tempGuessVec;
}

LMMat* SJLMSpectrumFitTask::getTransposedJacobian(LMVec& beta){
    float* cpuGuessVec = this->guessVec->getCPUVec();

    //calculate nnz
    float* featParams;
    int* rowPtr = new int[this->spec_len];

    for(int i = 0 ; i < nFeatures; i++){
        featParams = cpuGuessVec + i*8;
            //doppler width
        double gammaD = gaas::VoigtLineshape::dopplerHWHM(featParams[0], molarMass, tempK);
        // bounds
        double maxHW = gaas::VoigtLineshape::floatMax(gammaD, gaas::VoigtLineshape::floatMax(featParams[1], featParams[2]));
        double minWavenum = featParams[0] - maxHW * WAVENUM_WING;
        double maxWavenum = featParams[0] + maxHW * WAVENUM_WING;
        int minInd = gaas::VoigtLineshape::toWavenumIndex(startWavenum, wavenumStep, minWavenum) + 1; //the +1 makes this equivalent to hapi
        int maxInd = gaas::VoigtLineshape::toWavenumIndex(startWavenum, wavenumStep, maxWavenum);
        int delta = maxInd - minInd;
        
    }

    this->guessVec->destroyCPUVec(cpuGuessVec);
}

LMVec* SJLMSpectrumFitTask::getModel(LMVec* beta){

}

int SJLMSpectrumFitTask::getFeatWidth(float* featParams){
    
}
