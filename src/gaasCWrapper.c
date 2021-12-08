#include "Gaas.cuh"

/*extern "C"*/ void runSimFloat(double tempK, double pressureAtm, double conc, float* spectrumTarget, float* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, int isoNum, char* runID){

std::string gaasDir_s = std::string(gaasDir);
std::string moleculeID_s = std::string(moleculeID);
std::string isoNum_s = std::to_string(isoNum); //"_iso_"+isotopologueID+"_"
std::string runID_s = std::string(runID);

gaas::lineshapeSim::simHandler sh = gaas::lineshapeSim::simHandler(gaasDir_s+moleculeID_s+"_iso_"+isoNum_s+"_"+runID_s, gaasDir_s+moleculeID_s+"_iso_"+isoNum_s+"_tips.csv");

double molarMass=sh.molMassMap[moleculeID_s+isoNum_s]; 
double isotopeAbundance=sh.isoAbundanceMap[moleculeID_s+isoNum_s];
std::cout << "molarMass: " << molarMass << " iso Abundance: "<< isotopeAbundance <<"\n";

sh.runFloat(tempK,pressureAtm,conc, spectrumTarget, wavenumsTarget, wavenumRes, startWavenum, endWavenum, molarMass, isotopeAbundance);

}

/*extern "C"*/ void runSimDouble(double tempK, double pressureAtm, double conc, double* spectrumTarget, double* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, int isoNum, char* runID){

std::string gaasDir_s = std::string(gaasDir);
std::string moleculeID_s = std::string(moleculeID);
std::string isoNum_s = std::to_string(isoNum);
std::string runID_s = std::string(runID);

gaas::lineshapeSim::simHandler sh = gaas::lineshapeSim::simHandler(gaasDir_s+moleculeID_s+"_iso_"+isoNum_s+"_"+runID_s, gaasDir_s+moleculeID_s+"_iso_"+isoNum_s+"_tips.csv");

double molarMass=sh.molMassMap[moleculeID_s+isoNum_s]; 
double isotopeAbundance=sh.isoAbundanceMap[moleculeID_s+isoNum_s];
std::cout << "molarMass: " << molarMass << " iso Abundance: "<< isotopeAbundance <<"\n";

sh.runDouble(tempK,pressureAtm,conc, spectrumTarget, wavenumsTarget, wavenumRes, startWavenum, endWavenum, molarMass, isotopeAbundance);

}
