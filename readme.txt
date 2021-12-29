Build & install instructions:

make sure you have python3 installed

install the latest nvidia drivers for your GPU https://www.nvidia.com/Download/index.aspx

install microsoft visual studio build tools if you are using windows, this downloads the required windows c++ compiler

Then install the nvidia cuda toolkit at https://developer.nvidia.com/cuda-downloads

Then from there you should be able to compile the C/C++/Cuda code using gcc/g++/nvcc

The library can also be built as a python module to be directly swapped in as a much faster replacement of 
the hapi absorptionCoefficient_Voigt function. 

To build and install the python library refer to section 1.0 for Linux and 1.1 for Windows.

1.0: Linux python module build+install guide:

	First compile the gaasCWrapper shared object library by running (in a terminal):
	make all

	Then change into the pyInterface directory (dont try to run the build script without changing directories, it uses the current working directory):
	cd pyInterface

	Then build and install the python library:
	sudo python3 setup.py build
	sudo python3 setup.py install

	If you are using an IDE/virtual environment, you can copy the gaas.py file and the cpython .so file from 
	pyInterface/build/lib.linux... into the venv/lib/python3.7/site-packages folder to use it from the IDE. 

1.1: Windows python module build+install guide:
	open up an x64 Native Tools Command Prompt (if you dont have this, you need to download microsoft visual studio build tools)
	go to pyInterface folder
	python3 setup.py build
	python3 setup.py install --user
	
	if you need to rebuild the dynamic library ( GAAS.dll and GAAS.lib ):
	Download Microsoft Visual Studio.
	Create a new project and choose Cuda Runtime for project type.
	Name it GAAS
	Delete the project default file: kernel.cu
	In the solution explorer, right click GAAS and select Add->Existing Item... and add all of the files in src and include (from the git repo) to the project.	
	Then in the solution explorer, right click GAAS and select Properties and set the configuration to your current configuration.
	Under Properties->Configuration Properties->General, change Configuration Type to Dynamic Library (.dll)
	Under Properties->CUDA C/C++->Device, set the Code Generation Setting to one matching your GPU compute capability (ex. for compute capability 7.5, set the argument to "compute_75,sm_75")
	Under Properties->C/C++->General add the directory path to the include folder from the git repo to Additional Include Directories
	Under Properties->C/C++->PreProcessor add GAAS_EXPORT to the list of preprocessor definitions 
	Exit the properties tab, Right click on GAAS in the solution explorer and click build.
	Once its finished building, right click GAAS in the solution explorer and and select "Open folder in file explorer"
	Copy GAAS.dll and GAAS.lib to the git repo into the pyInterface folder (overwrite the existing files)
	
2.1: Supported GPU architectures:
	compute_35,sm_35
	compute_37,sm_37
	compute_50,sm_50
	compute_52,sm_52
	compute_61,sm_61
	compute_70,sm_70
	compute_75,sm_75
	compute_86,sm_86
