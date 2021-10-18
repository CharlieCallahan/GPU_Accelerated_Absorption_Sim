
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "Gaas.cuh"
#include "Stopwatch.hpp"
#include "file_io.hpp"


int main()
{
	gaas::lineshapeSim::simHandler sh = gaas::lineshapeSim::simHandler("/home/gputestbed/Desktop/WMS_Processing_V2_ARPAE/gaas/H2Oh2oTest", "/home/gputestbed/Desktop/WMS_Processing_V2_ARPAE/gaas/H2O_tips.csv");
	
	std::cout << "MEMORY TEST: check tops, repeatedly running absorption sim.\n";
	while(1){
		int wavenumRes = (6000-100)*1000;
		float* wavenums = new float[wavenumRes];
		float* spec = new float[wavenumRes];
		sh.runFloat(300, 1, .1, spec, wavenums, wavenumRes, 100, 6000);
		delete [] wavenums;
		delete [] spec;
	}
	
	//fio::save_to_bin_file((char*)spec, wavenumRes * sizeof(float), "/home/gputestbed/Desktop/GAAS/GAAS/Validation/output");
	
	
	return 0;
}


