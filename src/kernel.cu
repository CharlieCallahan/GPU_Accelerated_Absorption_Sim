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
 
#include <stdio.h>
#include <iostream>
#include <map>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Gaas.cuh"
#include "Stopwatch.hpp"
#include "file_io.hpp"

//Example C++ usage...

int main()
{
	gaas::lineshapeSim::simHandler sh = gaas::lineshapeSim::simHandler("gaas/H2Oh2oTest", "/gaas/H2O_tips.csv");
	int wavenumRes = (6000-100)*1000;
	float* wavenums = new float[wavenumRes];
	float* spec = new float[wavenumRes];
	sh.runFloat(300, 1, .1, spec, wavenums, wavenumRes, 100, 6000,18.01528,1);
	delete [] wavenums;
	delete [] spec;
	
	
	//fio::save_to_bin_file((char*)spec, wavenumRes * sizeof(float), "/home/gputestbed/Desktop/GAAS/GAAS/Validation/output");
	
	
	return 0;
}


