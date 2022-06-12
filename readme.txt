Build & install instructions:

Windows python module quick install guide:

	1. Install the latest Nvidia drivers for your GPU https://www.nvidia.com/Download/index.aspx

	2. Install Microsoft Visual Studio Build tools 2019 (The version must be 2017-2019, the newer versions arent yet compatible with the Nvidia libraries).

	3. Install the Nvidia Cuda toolkit at https://developer.nvidia.com/cuda-downloads , this downloads the GPU compute libraries.

	4. Run "x64 Native Tools Command Prompt for VS 2019"
	
	5. In the command prompt, move to the pyInterface directory using the command:
		cd {pyInterface directory path}
	
	6. Build the python module with the command: 
		python3 setup.py build
	
	7. Install the python module with the command:
		python3 setup.py install --user
	
	8. Now you should be able to use the GAAS api from your default python environment, if you want to use it from a virtual environment
	   you need to copy the .pyd file in pyInterface/build/lib.win-*/ into your virtual environment.

	9. The gaas.py file contains wrapper functions around the gaasAPI functions for ease of use. The functions in this file should be used in most cases.

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
	
Optional:
	if you need to rebuild the dynamic library ( GAAS.dll and GAAS.lib ):
	Download Microsoft Visual Studio 2017-2019 (these are the only versions compatible with CUDA).
	Open an x64 Native Tools Command Prompt for Visual Studio 2019
	run compileDLL.bat from the command line (this builds GAAS.dll and GAAS.lib and copies them to the pyInterface folder)

Building and Installing Python module:
	open up an x64 Native Tools Command Prompt (if you dont have this, you need to download Microsoft Visual Studio build tools)
	go to pyInterface folder
	python3 setup.py build
	python3 setup.py install --user
	
2.1: Supported GPU architectures:
	compute_35,sm_35
	compute_37,sm_37
	compute_50,sm_50
	compute_52,sm_52
	compute_61,sm_61
	compute_70,sm_70
	compute_75,sm_75
	compute_86,sm_86
