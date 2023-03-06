/**
 * @file SJLMASpectralFitter.hpp
 * @author Charlie Callahan (chca7857@colorado.edu)
 * @brief GPU based Sparse Jacobian Levenberg Marquardt spectral fitting routines
 * @version 0.1
 * @date 2023-01-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef SJLMASPECTRALFITTER_HPP
#define SJLMASPECTRALFITTER_HPP
#include "Lma.hpp"
#include "LmaCuda.cuh"
#include "inputStructs.hpp"


class SJLMSpectrumFitTask : public LMATask{

    /**
     * @brief Construct a new SJLMSpectrumFitTask object, pass in the absorption spectrum
     * 
     * @param absSpectrum 
     * @param spec_len 
     */
    SJLMSpectrumFitTask(float* absSpectrum, 
                        double startWavenum,
                        double endWavenum,
                        double wavenumStep,
                        int spec_len,
                        int nFeatures,
                        float tempK,
                        float molarMass,
                        featureDataHTP* guesses);

    LMMat* getTransposedJacobian(LMVec& beta) override;

    LMVec* getModel(LMVec* beta) override;

    CudaLMVec* specVec;
    CudaLMVec* guessVec;
    int nFeatures;
    float tempK;
    float molarMass;
    double startWavenum;
    double endWavenum;
    double wavenumStep;
    int spec_len;
    
    int getFeatWidth(float* featParams);
    
};

#endif /* SJLMASPECTRALFITTER_HPP */
