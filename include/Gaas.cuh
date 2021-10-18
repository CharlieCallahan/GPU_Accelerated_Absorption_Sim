//  Created by Charles callahan on 9/25/21.
//  Copyright © 2021 Charles callahan. All rights reserved.

// voigt profile functions (everything in namespace faadeeva) is copied over
// from http://ab-initio.mit.edu/Faddeeva and modified to 
// compile as a cuda kernel.

//faadeeva MIT Lisence

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
#include "Stopwatch.hpp"

#ifdef __cplusplus
extern "C" {
#endif

//#ifdef __INTELLISENSE__
//#include "intellisense_cuda_intrinsics.h"
//#endif

//#include <complex>

//Constants
#define ROOT2 1.41421356237
#define ROOT2PI 2.50662827463
#define LOG2 0.69314718056
#define SPEED_OF_LIGHT 299792458 //m s-1
#define AVOGADROS_NUM 6.02214129e23 //mol-1
#define BOLTZMANN_CONST 1.38064852e-23 //m2 kg s-2 K-1
#define CBOLTZ_CGS 1.380648813E-16 //CGS units
#define WAVENUM_WING 50 //how many halfwidths to compute around center of voigt function
#define DBL_EPSILON      2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0

//From MIT Implementation of Fadeeva function
typedef cuDoubleComplex cmplx;
#  define C(a,b) cmplx({a,b})
__host__ __device__ inline cmplx cuPolar(const double &magnitude, const double &angle) { return C(magnitude*cos(angle), magnitude*sin(angle)); }
__host__ __device__ inline cmplx cuExp(const cmplx& z) { return cuPolar(exp(cuCreal(z)), cuCimag(z)); }

#  define Inf INFINITY //redefines macros from MITs faadeeva to be compatible with cuda
#  define NaN NAN
#  define cexp(z) cuExp(z)
#  define creal(z) cuCreal(z)
#  define cimag(z) cuCimag(z)
#  define cpolar(r,t) cuPolar(r,t)


//__device__ double atomicAdd(double* addr, double val);
namespace gaas {

	void checkCudaErrors(cudaError_t& errorCode, std::string errorMessage);

	//Fadeeva subroutines
	namespace faddeeva {
		//__device__ std::complex<double> erfcx(std::complex<double> z, double relerr = 0);
		__device__ double erfcx_y100(double y100);
		__device__ double erfcx(double x); // special case for real x
		__device__ static inline double sqr(double x) { return x * x; }
		__device__ static inline double sinh_taylor(double x) {
			return x * (1 + (x*x) * (0.1666666666666666666667
				+ 0.00833333333333333333333 * (x*x)));
		}
		__device__ static inline double sinc(double x, double sinx) {
			return fabs(x) < 1e-4 ? 1 - (0.1666666666666666666667)*x*x : sinx / x;
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

	namespace lineshapeSim{

		struct featureData { //absorption feature database entry. data needed to calculate lineshape
			double transWavenum = 0;
			double nAir = 0;
			double gammaAir = 1; //these need to be non-zero so that features can be added to pad the input to be an integer # of cuda warps
			double gammaSelf = 1; //and the features wont have a division by 0 when being calculated
			double refStrength = 0;
			double ePrimePrime = 0;
			double deltaAir = 0;
			void print();
		};

		class PartitionSum {
		public:
			PartitionSum(std::string filename); //loads Psum data from csv file
			~PartitionSum() { delete[] pSumData; delete[] tempData; }
			double at(double tempK); //returns pSum data at temp in kelvin
		private:
			double* pSumData;
			double* tempData;
			int count; //Number of elements in Data arrays
		};

		struct simHandler { //handles simulation
						    //contains host code to call gaas accelerator
						    //holds feature database
			simHandler(std::string databaseFilename, std::string tipsFilename); 
			~simHandler() { delete[] featDatabase; delete tips; cudaError_t cudaStatus = cudaDeviceReset(); gaas::checkCudaErrors(cudaStatus, "Cuda device reset failed."); }
			//runs simulation on GPU, needs pre allocated arrays for spectrum target and wavenums target
			//runs with float precision
			void runFloat(double tempK, double pressureAtm, double conc, float* spectrumTarget, float* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum);
			 //double atomic add required for this is only available in cuda arch > 6.0 gpus
			void runDouble(double tempK, double pressureAtm, double conc, double* spectrumTarget, double* wavenumsTarget, int wavenumRes, double startWavenum, double endWavenum);
			
			featureData* featDatabase;
			int absFeatCount; //number of features in database
			PartitionSum* tips;

			//cuda params: these should be changeable through python interface
			int cudaThreadsPerBlock = 128; //may need to change this for different GPUs
			int cudaDevice = 0;

		private:
			void loadFeatDatabase(std::string databaseFilename);
		};

		__device__ double dopplerHWHM(double transWavenum, double molarMass, double tempKelvin);
		__device__ double lorentzianHWHM(double tempK, double pressureAtm, double pSelf,
			double nAir, double gammaAir, double gammaSelf);
		__device__ double lineStrength(double pSumT, double pSumTref, double refStrength, double ePrimePrime, double tempK, double transWavenum);

		__device__ inline int toWavenumIndex(double startWavenum, double wavenumStep, double wavenumInput);

		//main function for simulating lineshape.
		__global__ void lineshapeFloat(float* wavenums, featureData* database, float* output, double tempK, double pressAtm, double conc, double tipsRef, double tipsTemp, double startWavenum, double wavenumStep, int wavenumCount, double molarMass, double isotopeAbundance, int threadsPerBlock);
		//double atomic add required for this is only available in cuda arch > 6.0 gpus
		__global__ void lineshapeDouble(double* wavenums, featureData* database, double* output, double tempK, double pressAtm, double conc, double tipsRef, double tipsTemp, double startWavenum, double wavenumStep, int wavenumCount, double molarMass, double isotopeAbundance, int threadsPerBlock);
		
		

	}

	//fadeeva function
	__device__ cuDoubleComplex w(cuDoubleComplex z, double relerr = 0);

	__device__ double voigtSingle(double& x, double& gaussianHWHM, double& lorentzHWHM);
	
	__global__ void voigtTest(double* output, double* input);
}

#endif

#ifdef __cplusplus
}
#endif
