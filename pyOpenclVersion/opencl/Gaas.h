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

#ifndef GAAS_HPP
#define GAAS_HPP

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#endif

#ifdef NOFP64
	#define double float
#else
	typedef double real_t;
#endif
#include "clComplex.h"

// Constants
#define ROOT2 1.41421356237
#define ROOT2PI 2.50662827463
#define LOG2 0.69314718056
#define SPEED_OF_LIGHT 299792458			// m s-1
#define AVOGADROS_NUM 6.02214129e23			// mol-1
#define BOLTZMANN_CONST 1.38064852e-23		// m2 kg s-2 K-1
#define CBOLTZ_CGS 1.380648813E-16			// CGS units
#define WAVENUM_WING 50						// how many halfwidths to compute around center of voigt function
#define SQRT_PI 1.77245385091

// From MIT Implementation of Fadeeva function
typedef struct clComplex cmplx;

#define C(a, b) make_clComplex(a, b)

inline cmplx cuPolar(const double magnitude, const double angle)
{
	return C(magnitude * cos(angle), magnitude * sin(angle));
}

inline cmplx clExp(const cmplx z) { return clPolar(exp(clReal(z)), clImag(z)); }

#define Inf INFINITY // redefines macros from MITs faadeeva to be compatible with cuda/opencl
#define NaN NAN
#define cexp(z) clExp(z)
#define creal(z) clReal(z)
#define cimag(z) clImag(z)
#define cpolar(r, t) clPolar(r, t)

double erfcx_y100(double y100);

double erfcx(double x); // special case for real x

double sqr(double x) { return x * x; }

double sinh_taylor(double x)
{
	return x * (1 + (x * x) * (0.1666666666666666666667 + 0.00833333333333333333333 * (x * x)));
}

double sinc(double x, double sinx)
{
	return fabs(x) < 1e-4 ? 1 - (0.1666666666666666666667) * x * x : sinx / x;
}

double w_im(double x); // special-case code for Im[w(x)] of real x

double w_im_y100(double y100, double x);

#ifdef DOUBLE_SUPPORT_AVAILABLE

__constant double expa2n2[] = {
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

#endif //DOUBLE_SUPPORT_AVAILABLE

struct featureDataVoigt
{ // absorption feature database entry. data needed to calculate lineshape
	double transWavenum;
	double nAir;
	double gammaAir;  // these need to be non-zero so that features can be added to pad the input to be an integer # of cuda warps
	double gammaSelf; // and the features wont have a division by 0 when being calculated
	double refStrength;
	double ePrimePrime;
	double deltaAir;
};

double dopplerHWHM(double transWavenum, double molarMass, double tempKelvin);

double lorentzianHWHM(double tempK, double pressureAtm, double pSelf,
								 double nAir, double gammaAir, double gammaSelf);

double lineStrength(double pSumT, double pSumTref, double refStrength, double ePrimePrime, double tempK, double transWavenum);

inline int toWavenumIndex(double startWavenum, double wavenumStep, double wavenumInput);

// main function for simulating lineshape.
__kernel void lineshapeVoigt(__global const double *wavenums,
							__global const struct featureDataVoigt *database, 
							__global double *output, 
							double tempK, 
							double pressAtm, 
							double conc, 
							double tipsRef, 
							double tipsTemp, 
							double startWavenum, 
							double wavenumStep, 
							int wavenumCount, 
							double molarMass, 
							double isotopeAbundance);

// return the larger of the two values
double doubleMax(double f1, double f2);

struct featureDataHTP
{ //absorption feature database for HTP profile
    double linecenter; //line center (wavenumber), 32b double only has ~7 decimals of precision, too low to accurately position line
    double Gam0; //Gamma0
    double Gam2; //Gamma2
    double Delta0; //shift0
    double Delta2; //shift2
    double anuVC; //nuVC
    double eta; //eta
    double lineIntensity; //line intensity
};

// __kernel void lineshapeHTP(__global double *wavenums, 
// 							__global struct featureDataHTP *database, 
// 							__global double *output, 
// 							double tempK, 
// 							double startWavenum, 
// 							double wavenumStep, 
// 							int wavenumCount, 
// 							double molarMass, 
// 							int threadsPerBlock);

// fadeeva function
struct clComplex w(struct clComplex z);

double voigtSingle(double x, double gaussianHWHM, double lorentzHWHM);

// void atomic_add_global(volatile __global double *source, const double operand);

// void atomic_add_f(volatile global float* addr, const float val);

void __attribute__((always_inline)) atomic_add_f(volatile global float* addr, const float val) {
    union {
        uint  u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        next.f32 = (expected.f32=current.f32)+val; // ...*val for atomic_mul_f()
        current.u32 = atomic_cmpxchg((volatile global uint*)addr, expected.u32, next.u32);
    } while(current.u32!=expected.u32);
}

#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
void __attribute__((always_inline)) atomic_add_d(volatile global double* addr, const double val) {
    union {
        ulong  u64;
        double f64;
    } next, expected, current;
    current.f64 = *addr;
    do {
        next.f64 = (expected.f64=current.f64)+val; // ...*val for atomic_mul_d()
        current.u64 = atom_cmpxchg((volatile global ulong*)addr, expected.u64, next.u64);
    } while(current.u64!=expected.u64);
}
#endif

double doubleMax(double f1, double f2);

__kernel void voigtTest(__global double *output, __global const double *input);

#endif /* GAAS_HPP */
