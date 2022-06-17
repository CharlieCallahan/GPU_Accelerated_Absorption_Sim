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

	9. Run example.py with the command "python3 ./example.py" in powershell to confirm that everything was installed correctly. Note that this
	requires matplotlib and numpy to be installed in the python environment.
	
Linux python module quick install guide:
	1. Install the latest Nvidia drivers for your GPU https://www.nvidia.com/Download/index.aspx
	
	2. Install the Nvidia Cuda toolkit at https://developer.nvidia.com/cuda-downloads , this downloads the GPU compute libraries.
	
	3. Open a terminal window and go to the pyInterface folder with:
		cd {pyInterface directory path}
	
	6. Build the python module with the command: 
		sudo python3 setup.py build
	
	7. Install the python module with the command:
		sudo python3 setup.py install
		
	8. If you are using an IDE/virtual environment, you can copy the gaas.py file and the cpython .so file from 
	pyInterface/build/lib.linux... into the venv/lib/python3.7/site-packages folder to use it from the IDE. 
	
	9. Run example.py with the command "python3 ./example.py" in powershell to confirm that everything was installed correctly. Note that this
	requires matplotlib and numpy to be installed in the python environment.
	
===================================================================================================

If you need to rebuild the .so (linux) or .dll (Windows) refer to section 1.0 for Linux or 1.1 for Windows.

1.0: Linux python module build+install guide:
	
	1. Install the latest Nvidia drivers for your GPU https://www.nvidia.com/Download/index.aspx
	
	2. Install the Nvidia Cuda toolkit at https://developer.nvidia.com/cuda-downloads , this downloads the GPU compute libraries.

	3. Open a terminal window and move to GPU_Accelerated_Absorption_Sim directory with:
		cd {GPU_Accelerated_Absorption_Sim path}
	
	4. Compile the gaasCWrapper shared object library by running (in a terminal):
		make all

	5. Then move into the pyInterface directory (dont try to run the build script without changing directories, it uses the current working directory):
		cd pyInterface

	6. Then build and install the python library:
		sudo python3 setup.py build
		sudo python3 setup.py install

1.1: Windows dll build guide:

	1. Install the latest Nvidia drivers for your GPU https://www.nvidia.com/Download/index.aspx

	2. Install Microsoft Visual Studio Build tools 2019 (The version must be 2017-2019, the newer versions arent yet compatible with the Nvidia libraries).

	3. Install the Nvidia Cuda toolkit at https://developer.nvidia.com/cuda-downloads , this downloads the GPU compute libraries.

	4. Run "x64 Native Tools Command Prompt for VS 2019"
	
	5. run compileDLL.bat from the command line (this builds GAAS.dll and GAAS.lib and copies them to the pyInterface folder)
	
	6. open up an x64 Native Tools Command Prompt (if you dont have this, you need to download Microsoft Visual Studio build tools)
	
	5. Then move into the pyInterface directory (dont try to run the build script without changing directories, it uses the current working directory):
		cd pyInterface

	6. Then build and install the python library:
		sudo python3 setup.py build
		sudo python3 setup.py install
	
2.1: Supported GPU architectures:
	compute_35,sm_35
	compute_37,sm_37
	compute_50,sm_50
	compute_52,sm_52
	compute_61,sm_61
	compute_70,sm_70
	compute_75,sm_75
	compute_86,sm_86
	
