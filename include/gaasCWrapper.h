

void runSimFloat(double tempK, double pressureAtm, double conc, float* spectrumTarget, float* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, char* runID); //runs simulation with 32 bit float precision, this works on older GPU architectures with slightly more error due to numerical precision

void runSimDouble(double tempK, double pressureAtm, double conc, double* spectrumTarget, double* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, char* runID); //runs simulation with 32 bit float precision, this works on older GPU architectures with slightly more error due to numerical precision

