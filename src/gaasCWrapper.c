#include "Gaas.cuh"

extern "C" void runSimFloat(double tempK, double pressureAtm, double conc, float* spectrumTarget, float* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, char* runID){

std::string gaasDir_s = std::string(gaasDir);
std::string moleculeID_s = std::string(moleculeID);
std::string runID_s = std::string(runID);

gaas::lineshapeSim::simHandler sh = gaas::lineshapeSim::simHandler(gaasDir_s+moleculeID_s+runID_s, gaasDir_s+moleculeID_s+std::string("_tips.csv"));
sh.runFloat(tempK,pressureAtm,conc, spectrumTarget, wavenumsTarget, wavenumRes, startWavenum, endWavenum);

}

extern "C" void runSimDouble(double tempK, double pressureAtm, double conc, double* spectrumTarget, double* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, char* runID){

std::string gaasDir_s = std::string(gaasDir);
std::string moleculeID_s = std::string(moleculeID);
std::string runID_s = std::string(runID);

gaas::lineshapeSim::simHandler sh = gaas::lineshapeSim::simHandler(gaasDir_s+moleculeID_s+runID_s, gaasDir_s+moleculeID_s+std::string("_tips.csv"));
sh.runDouble(tempK,pressureAtm,conc, spectrumTarget, wavenumsTarget, wavenumRes, startWavenum, endWavenum);

}
