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

#include "Gaas.cuh"

#if !(defined(WIN64) || defined(WIN32))
extern "C"
{
#endif

#include "gaasCWrapper.h"

    /*extern "C"*/ void runSimFloat(double tempK, double pressureAtm, double conc, float *spectrumTarget, double *wavenumsTarget, double wavenumStep, double startWavenum, double endWavenum, char *gaasDir, char *moleculeID, int isoNum, char *runID)
    {

        std::string gaasDir_s = std::string(gaasDir);
        std::string moleculeID_s = std::string(moleculeID);
        std::string isoNum_s = std::to_string(isoNum); //"_iso_"+isotopologueID+"_"
        std::string runID_s = std::string(runID);

        gaas::lineshapeSim::simHandler sh = gaas::lineshapeSim::simHandler(gaasDir_s + moleculeID_s + "_iso_" + isoNum_s + "_" + runID_s, gaasDir_s + moleculeID_s + "_iso_" + isoNum_s + "_tips.csv");

        double molarMass = sh.molMassMap[moleculeID_s + isoNum_s];
        double isotopeAbundance = sh.isoAbundanceMap[moleculeID_s + isoNum_s];
#ifndef SILENT
        std::cout << "molarMass: " << molarMass << " iso Abundance: " << isotopeAbundance << "\n";
#endif
        sh.runFloat(tempK, pressureAtm, conc, spectrumTarget, wavenumsTarget, wavenumStep, startWavenum, endWavenum, molarMass, isotopeAbundance);
    }

    /*extern "C"*/ void runSimDouble(double tempK, double pressureAtm, double conc, double *spectrumTarget, double *wavenumsTarget, double wavenumStep, double startWavenum, double endWavenum, char *gaasDir, char *moleculeID, int isoNum, char *runID)
    {

        std::string gaasDir_s = std::string(gaasDir);
        std::string moleculeID_s = std::string(moleculeID);
        std::string isoNum_s = std::to_string(isoNum);
        std::string runID_s = std::string(runID);

        gaas::lineshapeSim::simHandler sh = gaas::lineshapeSim::simHandler(gaasDir_s + moleculeID_s + "_iso_" + isoNum_s + "_" + runID_s, gaasDir_s + moleculeID_s + "_iso_" + isoNum_s + "_tips.csv");

        double molarMass = sh.molMassMap[moleculeID_s + isoNum_s];
        double isotopeAbundance = sh.isoAbundanceMap[moleculeID_s + isoNum_s];
#ifndef SILENT
        std::cout << "molarMass: " << molarMass << " iso Abundance: " << isotopeAbundance << "\n";
#endif
        sh.runDouble(tempK, pressureAtm, conc, spectrumTarget, wavenumsTarget, wavenumStep, startWavenum, endWavenum, molarMass, isotopeAbundance);
    }

#if !(defined(WIN64) || defined(WIN32))
}
#endif
