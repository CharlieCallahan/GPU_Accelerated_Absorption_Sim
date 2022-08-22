@ECHO ON
set nvccFlags=-Iinclude -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -lcudart_static -lcuda -DWIN64 -DGAAS_EXPORT -DSILENT
nvcc -rdc=true -g -o GAAS.dll --shared src\Gaas.cu src\file_io.cpp src\gaasCWrapper.cpp src\Stopwatch.cpp %nvccFlags%
copy GAAS.dll pyInterface\GAAS.dll
copy GAAS.lib pyInterface\GAAS.lib
del *.obj 
del *.pdb
del *.exp
del GAAS.dll GAAS.lib
PAUSE 