#ifndef CLCOMPLEX_HPP
#define CLCOMPLEX_HPP

struct clComplex{
    // clComplex(double real, double cmplx){re=real;cmp=cmplx;}
    double re;
    double cmp;
};

struct clComplex make_clComplex(double real, double cmplx){
    struct clComplex out;
    out.re = real;
    out.cmp = cmplx;
    return out;
}

double clReal(struct clComplex z){
    return z.re;
}

double clImag(struct clComplex z){
    return z.cmp;
}

struct clComplex clPolar(double r, double t){
    return make_clComplex(r * cos(t), r * sin(t));
}

struct clComplex clMul(struct clComplex x, struct clComplex y){
    struct clComplex prod;
    prod = make_clComplex((clReal(x) * clReal(y)) - 
                    (clImag(x) * clImag(y)),
                    (clReal(x) * clImag(y)) + 
                    (clImag(x) * clReal(y)));
    return prod;
}

struct clComplex clDiv(struct clComplex x,struct clComplex y)
{
    struct clComplex quot;
    double s = (fabs(clReal(y))) + (fabs(clImag(y)));
    double oos = 1.0 / s;
    double ars = clReal(x) * oos;
    double ais = clImag(x) * oos;
    double brs = clReal(y) * oos;
    double bis = clImag(y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    quot = make_clComplex (((ars * brs) + (ais * bis)) * oos,
                                 ((ais * brs) - (ars * bis)) * oos);
    return quot;
}

struct clComplex clSub(struct clComplex x, struct clComplex y) {
    return make_clComplex(clReal(x) - clReal(y), clImag(x) - clImag(y));
}

struct clComplex clAdd(struct clComplex x, struct clComplex y){
    return make_clComplex(clReal(x) + clReal(y), clImag(x) + clImag(y));
}

//complex square root adapted from head/lib/msun/src/s_csqrt.c from freeBSD

/*-
 * Copyright (c) 2007 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */


/* We risk spurious overflow for components >= DBL_MAX / (1 + sqrt(2)). */
#define	THRESH	0x1.a827999fcef32p+1022

struct clComplex clSqrt(struct clComplex z)
{
	struct clComplex result;
	double a, b;
	double t;
	int scale;

	a = clReal(z);
	b = clImag(z);

	/* Handle special cases. */
	if (a == 0.0f && b == 0.0f)
		return make_clComplex(0, b);
	// if (isinf(b))
	// 	return (make_clComplex(INFINITY, b));
	// if (isnan(a)) {
	// 	t = (b - b) / (b - b);	/* raise invalid if b is not a NaN */
	// 	return (cpack(a, t));	/* return NaN + NaN i */
	// }
	// if (isinf(a)) {
	// 	/*
	// 	 * csqrt(inf + NaN i)  = inf +  NaN i
	// 	 * csqrt(inf + y i)    = inf +  0 i
	// 	 * csqrt(-inf + NaN i) = NaN +- inf i
	// 	 * csqrt(-inf + y i)   = 0   +  inf i
	// 	 */
	// 	if (signbit(a))
	// 		return (cpack(fabs(b - b), copysign(a, b)));
	// 	else
	// 		return (cpack(a, copysign(b - b, b)));
	// }
	/*
	 * The remaining special case (b is NaN) is handled just fine by
	 * the normal code path below.
	 */

	/* Scale to avoid overflow. */
	if (fabs(a) >= THRESH || fabs(b) >= THRESH) {
		a *= 0.25;
		b *= 0.25;
		scale = 1;
	} else {
		scale = 0;
	}

	/* Algorithm 312, CACM vol 10, Oct 1967. */
	if (a >= 0) {
		t = sqrt((a + sqrt(a*a + b*b)) * 0.5);
		result = make_clComplex(t, b / (2 * t));
	} else {
		t = sqrt((-a + sqrt(a*a + b*b)) * 0.5);
		result = make_clComplex(fabs(b) / (2 * t), copysign(t, b));
	}

	/* Rescale. */
	if (scale)
		return clMul(result,make_clComplex(2.0,0));
	else
		return result;
}

#endif /* CLCOMPLEX_HPP */
