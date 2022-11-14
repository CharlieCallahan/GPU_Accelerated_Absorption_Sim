/* Copyright (c) 2021 Charlie Callahan
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "inputStructs.hpp"

#if defined(WIN64) || defined(WIN32)

#ifdef GAAS_EXPORT

#define FUNC_PREPEND extern "C" __declspec(dllexport)
#else
#define FUNC_PREPEND __declspec(dllimport)
#endif

#else // linux mac etc (linking to shared object instead of dll)
#define FUNC_PREPEND
#endif

FUNC_PREPEND void simVoigt(double tempK, double pressureAtm, double conc, float *spectrumTarget, double *wavenumsTarget, double wavenumStep, double startWavenum, double endWavenum, char *gaasDir, char *moleculeID, int isoNum, char *runID); // runs simulation with 32 bit float precision, this works on older GPU architectures with slightly more error due to numerical precision

/**
 * @brief Runs a HT profile simulation over the wavenumber domain (startWavenum to endWavenum with resolution given by wavenumStep)
 *        Calculates the profile for each absorption profile in the features array in parallel and adds them to the output array (spectrumTarget).
 * 
 * @param features 
 * @param numFeatures 
 * @param spectrumTarget 
 * @param wavenumsTarget 
 * @param wavenumStep 
 * @param startWavenum 
 * @param endWavenum 
 * @return FUNC_PREPEND 
 */

FUNC_PREPEND void simHTP(struct featureDataHTP *features, int numFeatures, float tempK, float molarMass, float *spectrumTarget, double *wavenumsTarget, double wavenumStep, double startWavenum, double endWavenum);
