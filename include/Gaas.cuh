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

// voigt profile functions (everything in namespace faadeeva) is copied over
// from http://ab-initio.mit.edu/Faddeeva and modified to
// compile as a cuda kernel.

// faadeeva MIT License

/* Copyright (c) 2012 Massachusetts Institute of Technology
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

#ifndef Gaas_cuh
#define Gaas_cuh
#define CUDA_THREADS_PER_BLOCK 128
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <cmath>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "Stopwatch.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

// Constants
#define ROOT2 1.41421356237
#define ROOT2PI 2.50662827463
#define LOG2 0.69314718056
#define SPEED_OF_LIGHT 299792458			// m s-1
#define AVOGADROS_NUM 6.02214129e23			// mol-1
#define BOLTZMANN_CONST 1.38064852e-23		// m2 kg s-2 K-1
#define CBOLTZ_CGS 1.380648813E-16			// CGS units
#define WAVENUM_WING 50						// how many halfwidths to compute around center of voigt function
#define DBL_EPSILON 2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0

	// From MIT Implementation of Fadeeva function
	typedef cuDoubleComplex cmplx;
#define C(a, b) cmplx({a, b})
	__host__ __device__ inline cmplx cuPolar(const double &magnitude, const double &angle)
	{
		return C(magnitude * cos(angle), magnitude * sin(angle));
	}
	__host__ __device__ inline cmplx cuExp(const cmplx &z) { return cuPolar(exp(cuCreal(z)), cuCimag(z)); }

#define Inf INFINITY // redefines macros from MITs faadeeva to be compatible with cuda
#define NaN NAN
#define cexp(z) cuExp(z)
#define creal(z) cuCreal(z)
#define cimag(z) cuCimag(z)
#define cpolar(r, t) cuPolar(r, t)

	namespace gaas
	{

		void checkCudaErrors(cudaError_t &errorCode, std::string errorMessage);

		// Fadeeva subroutines
		namespace faddeeva
		{
			__device__ double erfcx_y100(double y100);

			__device__ double erfcx(double x); // special case for real x

			__device__ static inline double sqr(double x) { return x * x; }

			__device__ static inline double sinh_taylor(double x)
			{
				return x * (1 + (x * x) * (0.1666666666666666666667 + 0.00833333333333333333333 * (x * x)));
			}

			__device__ static inline double sinc(double x, double sinx)
			{
				return fabs(x) < 1e-4 ? 1 - (0.1666666666666666666667) * x * x : sinx / x;
			}

			__device__ double w_im(double x); // special-case code for Im[w(x)] of real x

			__device__ double w_im_y100(double y100, double x);

			__device__ static const double expa2n2[] = {
				7.64405281671221563e-01,
				3.41424527166548425e-01,
				8.91072646929412548e-02,
				1.35887299055460086e-02,
				1.21085455253437481e-03,
				6.30452613933449404e-05,
				1.91805156577114683e-06,
				3.40969447714832381e-08,
				3.54175089099469393e-10,
				2.14965079583260682e-12,
				7.62368911833724354e-15,
				1.57982797110681093e-17,
				1.91294189103582677e-20,
				1.35344656764205340e-23,
				5.59535712428588720e-27,
				1.35164257972401769e-30,
				1.90784582843501167e-34,
				1.57351920291442930e-38,
				7.58312432328032845e-43,
				2.13536275438697082e-47,
				3.51352063787195769e-52,
				3.37800830266396920e-57,
				1.89769439468301000e-62,
				6.22929926072668851e-68,
				1.19481172006938722e-73,
				1.33908181133005953e-79,
				8.76924303483223939e-86,
				3.35555576166254986e-92,
				7.50264110688173024e-99,
				9.80192200745410268e-106,
				7.48265412822268959e-113,
				3.33770122566809425e-120,
				8.69934598159861140e-128,
				1.32486951484088852e-135,
				1.17898144201315253e-143,
				6.13039120236180012e-152,
				1.86258785950822098e-160,
				3.30668408201432783e-169,
				3.43017280887946235e-178,
				2.07915397775808219e-187,
				7.36384545323984966e-197,
				1.52394760394085741e-206,
				1.84281935046532100e-216,
				1.30209553802992923e-226,
				5.37588903521080531e-237,
				1.29689584599763145e-247,
				1.82813078022866562e-258,
				1.50576355348684241e-269,
				7.24692320799294194e-281,
				2.03797051314726829e-292,
				3.34880215927873807e-304,
				0.0 // underflow (also prevents reads past array end, below)
			};

		}

		namespace lineshapeSim
		{

			struct featureData
			{ // absorption feature database entry. data needed to calculate lineshape
				double transWavenum = 0;
				double nAir = 0;
				double gammaAir = 1;  // these need to be non-zero so that features can be added to pad the input to be an integer # of cuda warps
				double gammaSelf = 1; // and the features wont have a division by 0 when being calculated
				double refStrength = 0;
				double ePrimePrime = 0;
				double deltaAir = 0;
				void print();
			};

			class PartitionSum
			{
			public:
				PartitionSum(std::string filename); // loads Psum data from csv file

				~PartitionSum()
				{
					delete[] pSumData;
					delete[] tempData;
				}

				double at(double tempK); // returns pSum data at temp in kelvin

			private:
				double *pSumData;
				double *tempData;
				int count;
			};

			struct simHandler
			{ // handles simulation
			  // contains host code to call gaas accelerator
			  // holds feature database
				simHandler(std::string databaseFilename, std::string tipsFilename);

				~simHandler()
				{
					delete[] featDatabase;
					delete tips;
					cudaError_t cudaStatus = cudaDeviceReset();
					gaas::checkCudaErrors(cudaStatus, "Cuda device reset failed.");
				}

				// runs simulation on GPU, needs pre allocated arrays for spectrum target and wavenums target
				// runs with float precision
				void runFloat(double tempK, double pressureAtm, double conc, float *spectrumTarget, double *wavenumsTarget, double wavenumStep, double startWavenum, double endWavenum, double molarMass, double isotopeAbundance);

				// double atomic add required for this is only available in cuda arch > 6.0 gpus
				void runDouble(double tempK, double pressureAtm, double conc, double *spectrumTarget, double *wavenumsTarget, double wavenumStep, double startWavenum, double endWavenum, double molarMass, double isotopeAbundance);

				featureData *featDatabase;
				int absFeatCount; // number of features in database
				PartitionSum *tips;

				// maps string of molID+isoNum to molarMass ex 'H2O1 = 18.01528'
				// these are taken from https://hitran.org/media/molparam.txt
				std::map<std::string, double> molMassMap{{"H2O1", 18.010565}, {"H2O2", 20.014811}, {"H2O3", 19.01478}, {"H2O4", 19.01674}, {"H2O5", 21.020985}, {"H2O6", 20.020956}, {"H2O7", 20.022915}, {"CO21", 43.98983}, {"CO22", 44.993185}, {"CO23", 45.994076}, {"CO24", 44.994045}, {"CO25", 46.997431}, {"CO26", 45.9974}, {"CO27", 47.99832}, {"CO28", 46.998291}, {"CO29", 45.998262}, {"CO210", 49.001675}, {"CO211", 48.001646}, {"CO212", 47.001618}, {"O31", 47.984745}, {"O32", 49.988991}, {"O33", 49.988991}, {"O34", 48.98896}, {"O35", 48.98896}, {"N2O1", 44.001062}, {"N2O2", 44.998096}, {"N2O3", 44.998096}, {"N2O4", 46.005308}, {"N2O5", 45.005278}, {"CO1", 27.994915}, {"CO2", 28.99827}, {"CO3", 29.999161}, {"CO4", 28.99913}, {"CO5", 31.002516}, {"CO6", 30.002485}, {"CH41", 16.0313}, {"CH42", 17.034655}, {"CH43", 17.037475}, {"CH44", 18.04083}, {"O21", 31.98983}, {"O22", 33.994076}, {"O23", 32.994045}, {"NO1", 29.997989}, {"NO2", 30.995023}, {"NO3", 32.002234}, {"SO21", 63.961901}, {"SO22", 65.957695}, {"SO23", 64.961286}, {"SO24", 65.966146}, {"NO21", 45.992904}, {"NO22", 46.989938}, {"NH31", 17.026549}, {"NH32", 18.023583}, {"HNO31", 62.995644}, {"HNO32", 63.992678}, {"OH1", 17.00274}, {"OH2", 19.006986}, {"OH3", 18.008915}, {"HF1", 20.006229}, {"HF2", 21.012404}, {"HCl1", 35.976678}, {"HCl2", 37.973729}, {"HCl3", 36.982853}, {"HCl4", 38.979904}, {"HBr1", 79.92616}, {"HBr2", 81.924115}, {"HBr3", 80.932336}, {"HBr4", 82.930289}, {"HI1", 127.912297}, {"HI2", 128.918472}, {"ClO1", 50.963768}, {"ClO2", 52.960819}, {"OCS1", 59.966986}, {"OCS2", 61.96278}, {"OCS3", 60.970341}, {"OCS4", 60.966371}, {"OCS5", 61.971231}, {"OCS6", 62.966137}, {"H2CO1", 30.010565}, {"H2CO2", 31.01392}, {"H2CO3", 32.014811}, {"HOCl1", 51.971593}, {"HOCl2", 53.968644}, {"N21", 28.006148}, {"N22", 29.003182}, {"HCN1", 27.010899}, {"HCN2", 28.014254}, {"HCN3", 28.007933}, {"CH3Cl1", 49.992328}, {"CH3Cl2", 51.989379}, {"H2O21", 34.00548}, {"C2H21", 26.01565}, {"C2H22", 27.019005}, {"C2H23", 27.021825}, {"C2H61", 30.04695}, {"C2H62", 31.050305}, {"PH31", 33.997241}, {"COF21", 65.991722}, {"COF22", 66.995078}, {"SF61", 145.962494}, {"H2S1", 33.987721}, {"H2S2", 35.983515}, {"H2S3", 34.987105}, {"HCOOH1", 46.00548}, {"HO21", 32.997655}, {"O1", 15.994915}, {"ClONO21", 96.956672}, {"ClONO22", 98.953723}, {"NO+1", 29.997989}, {"HOBr1", 95.921076}, {"HOBr2", 97.91903}, {"C2H41", 28.0313}, {"C2H42", 29.034655}, {"CH3OH1", 32.026215}, {"CH3Br1", 93.941811}, {"CH3Br2", 95.939764}, {"CH3CN1", 41.026549}, {"CF41", 87.993616}, {"C4H21", 50.01565}, {"HC3N1", 51.010899}, {"H21", 2.01565}, {"H22", 3.021825}, {"CS1", 43.97207}, {"CS2", 45.967866}, {"CS3", 44.975425}, {"CS4", 44.971456}, {"SO31", 79.956815}, {"C2N21", 52.006148}, {"COCl21", 97.93262}, {"COCl22", 99.929672}, {"SO1", 47.966986}, {"SO2", 49.962782}, {"SO3", 49.971231}, {"CH3F1", 34.021878}, {"GeH41", 77.952479}, {"GeH42", 75.95338}, {"GeH43", 73.95555}, {"GeH44", 76.954764}, {"GeH45", 79.952703}, {"CS21", 75.94414}, {"CS22", 77.939936}, {"CS23", 76.943526}, {"CS24", 76.947495}, {"CH3I1", 141.927947}, {"NF31", 70.998286}};
				std::map<std::string, double> isoAbundanceMap{{"H2O1", 0.997317}, {"H2O2", 0.00199983}, {"H2O3", 0.000371884}, {"H2O4", 0.000310693}, {"H2O5", 6.23003e-07}, {"H2O6", 1.15853e-07}, {"H2O7", 2.41974e-08}, {"CO21", 0.984204}, {"CO22", 0.0110574}, {"CO23", 0.00394707}, {"CO24", 0.000733989}, {"CO25", 4.43446e-05}, {"CO26", 8.24623e-06}, {"CO27", 3.95734e-06}, {"CO28", 1.4718e-06}, {"CO29", 1.36847e-07}, {"CO210", 4.446e-08}, {"CO211", 1.65354e-08}, {"CO212", 1.53745e-09}, {"O31", 0.992901}, {"O32", 0.00398194}, {"O33", 0.00199097}, {"O34", 0.000740475}, {"O35", 0.000370237}, {"N2O1", 0.990333}, {"N2O2", 0.00364093}, {"N2O3", 0.00364093}, {"N2O4", 0.00198582}, {"N2O5", 0.00036928}, {"CO1", 0.986544}, {"CO2", 0.0110836}, {"CO3", 0.00197822}, {"CO4", 0.000367867}, {"CO5", 2.2225e-05}, {"CO6", 4.13292e-06}, {"CH41", 0.988274}, {"CH42", 0.0111031}, {"CH43", 0.000615751}, {"CH44", 6.91785e-06}, {"O21", 0.995262}, {"O22", 0.00399141}, {"O23", 0.000742235}, {"NO1", 0.993974}, {"NO2", 0.00365431}, {"NO3", 0.00199312}, {"SO21", 0.945678}, {"SO22", 0.0419503}, {"SO23", 0.00746446}, {"SO24", 0.00379256}, {"NO21", 0.991616}, {"NO22", 0.00364564}, {"NH31", 0.995872}, {"NH32", 0.00366129}, {"HNO31", 0.98911}, {"HNO32", 0.00363643}, {"OH1", 0.997473}, {"OH2", 0.00200014}, {"OH3", 0.000155371}, {"HF1", 0.999844}, {"HF2", 0.000155741}, {"HCl1", 0.757587}, {"HCl2", 0.242257}, {"HCl3", 0.000118005}, {"HCl4", 3.7735e-05}, {"HBr1", 0.506781}, {"HBr2", 0.493063}, {"HBr3", 7.89384e-05}, {"HBr4", 7.68016e-05}, {"HI1", 0.999844}, {"HI2", 0.000155741}, {"ClO1", 0.755908}, {"ClO2", 0.24172}, {"OCS1", 0.937395}, {"OCS2", 0.0415828}, {"OCS3", 0.0105315}, {"OCS4", 0.00739908}, {"OCS5", 0.00187967}, {"OCS6", 0.000467176}, {"H2CO1", 0.986237}, {"H2CO2", 0.0110802}, {"H2CO3", 0.00197761}, {"HOCl1", 0.75579}, {"HOCl2", 0.241683}, {"N21", 0.992687}, {"N22", 0.00729916}, {"HCN1", 0.985114}, {"HCN2", 0.0110676}, {"HCN3", 0.00362174}, {"CH3Cl1", 0.748937}, {"CH3Cl2", 0.239491}, {"H2O21", 0.994952}, {"C2H21", 0.977599}, {"C2H22", 0.0219663}, {"C2H23", 0.00030455}, {"C2H61", 0.97699}, {"C2H62", 0.0219526}, {"PH31", 0.999533}, {"COF21", 0.986544}, {"COF22", 0.0110837}, {"SF61", 0.95018}, {"H2S1", 0.949884}, {"H2S2", 0.0421369}, {"H2S3", 0.00749766}, {"HCOOH1", 0.983898}, {"HO21", 0.995107}, {"O1", 0.997628}, {"ClONO21", 0.74957}, {"ClONO22", 0.239694}, {"NO+1", 0.993974}, {"HOBr1", 0.505579}, {"HOBr2", 0.491894}, {"C2H41", 0.977294}, {"C2H42", 0.0219595}, {"CH3OH1", 0.98593}, {"CH3Br1", 0.500995}, {"CH3Br2", 0.487433}, {"CH3CN1", 0.973866}, {"CF41", 0.98889}, {"C4H21", 0.955998}, {"HC3N1", 0.963346}, {"H21", 0.999688}, {"H22", 0.000311432}, {"CS1", 0.939624}, {"CS2", 0.0416817}, {"CS3", 0.0105565}, {"CS4", 0.00741668}, {"SO31", 0.943434}, {"C2N21", 0.970752}, {"COCl21", 0.566392}, {"COCl22", 0.362235}, {"SO1", 0.947926}, {"SO2", 0.04205}, {"SO3", 0.00190079}, {"CH3F1", 0.988428}, {"GeH41", 0.365172}, {"GeH42", 0.274129}, {"GeH43", 0.205072}, {"GeH44", 0.0775517}, {"GeH45", 0.0775517}, {"CS21", 0.892811}, {"CS22", 0.0792103}, {"CS23", 0.0140944}, {"CS24", 0.0100306}, {"CH3I1", 0.988428}, {"NF31", 0.996337}};

				int cudaThreadsPerBlock = CUDA_THREADS_PER_BLOCK; // may need to change this for different GPUs
				int cudaDevice = 0;

			private:
				void loadFeatDatabase(std::string databaseFilename);
			};

			__device__ double dopplerHWHM(double transWavenum, double molarMass, double tempKelvin);

			__device__ double lorentzianHWHM(double tempK, double pressureAtm, double pSelf,
											 double nAir, double gammaAir, double gammaSelf);

			__device__ double lineStrength(double pSumT, double pSumTref, double refStrength, double ePrimePrime, double tempK, double transWavenum);

			__device__ inline int toWavenumIndex(double startWavenum, double wavenumStep, double wavenumInput);

			// main function for simulating lineshape.
			__global__ void lineshapeFloat(double *wavenums, featureData *database, float *output, double tempK, double pressAtm, double conc, double tipsRef, double tipsTemp, double startWavenum, double wavenumStep, int wavenumCount, double molarMass, double isotopeAbundance, int threadsPerBlock);
			// double atomic add required for this is only available in cuda arch > 6.0 gpus
			__global__ void lineshapeDouble(double *wavenums, featureData *database, double *output, double tempK, double pressAtm, double conc, double tipsRef, double tipsTemp, double startWavenum, double wavenumStep, int wavenumCount, double molarMass, double isotopeAbundance, int threadsPerBlock);

		}

		// fadeeva function
		__device__ cuDoubleComplex w(cuDoubleComplex z, double relerr = 0);

		__device__ double voigtSingle(double &x, double &gaussianHWHM, double &lorentzHWHM);

		__global__ void voigtTest(double *output, double *input);
	}

#ifdef __cplusplus
}
#endif
#endif