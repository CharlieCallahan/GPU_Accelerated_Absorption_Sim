#if defined(WIN64) || defined(WIN32)

#ifdef GAAS_EXPORT

#define FUNC_PREPEND extern "C" __declspec(dllexport)
#else
#define FUNC_PREPEND __declspec(dllimport)
#endif

#else //linux mac etc (linking to shared object instead of dll)
#define FUNC_PREPEND  
#endif

FUNC_PREPEND void runSimFloat(double tempK, double pressureAtm, double conc, float* spectrumTarget, float* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, int isoNum, char* runID); //runs simulation with 32 bit float precision, this works on older GPU architectures with slightly more error due to numerical precision

FUNC_PREPEND void runSimDouble(double tempK, double pressureAtm, double conc, double* spectrumTarget, double* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, int isoNum, char* runID); //runs simulation with 32 bit float precision, this works on older GPU architectures with slightly more error due to numerical precision

