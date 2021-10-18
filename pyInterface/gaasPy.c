#include <Python.h>
#include "gaasCWrapper.h"
#include <stdio.h>

static PyObject* runSimF32(PyObject* self, PyObject* args){
//ARGS: (double tempK, double pressureAtm, double conc, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, char* runID)
	double tempK;
	double pressureAtm;
	double conc;
	int wavenumRes;
	double startWavenum;
	double endWavenum;
	char* gaasDir;
	char* moleculeID;
	char* runID;
	
	
	if (!PyArg_ParseTuple(args, "dddiddsss", &tempK, &pressureAtm, &conc, &wavenumRes, &startWavenum, &endWavenum, &gaasDir, &moleculeID, &runID)){
		return NULL;
	}
	
	//float* spectrumTarget = new float [wavenumRes];
	//float* wavenumsTarget = new float [wavenumRes];
	
	float* spectrumTarget = (float*)malloc(wavenumRes*sizeof(float));//new float [wavenumRes];
	float* wavenumsTarget = (float*)malloc(wavenumRes*sizeof(float));
	
	runSimFloat(tempK, pressureAtm, conc, spectrumTarget, wavenumsTarget, wavenumRes, startWavenum, endWavenum, gaasDir, moleculeID, runID);
	
	PyObject* spectrumList = PyList_New(wavenumRes);
	PyObject* wavenumsList = PyList_New(wavenumRes);
	PyObject* tempVal;
	
	for(int i = 0; i < wavenumRes; i++){
	//add results to py lists for output
	tempVal = Py_BuildValue("f",spectrumTarget[i]);
	PyList_SetItem(spectrumList, i, tempVal);
	
	tempVal = Py_BuildValue("f",wavenumsTarget[i]);
	PyList_SetItem(wavenumsList, i, tempVal);
	}
	
	PyObject* output = PyTuple_New(2);
	if ( !(PyTuple_SetItem(output, 0, spectrumList) == 0))
		printf("Set output tuple 0 failed...");
	if ( !(PyTuple_SetItem(output, 1, wavenumsList) == 0))
		printf("Set output tuple 1 failed...");
		
	free(spectrumTarget);
	free(wavenumsTarget);
	return output;
}

//runs simulation with double point precision, requires cuda architecture > 6.0
static PyObject* runSimF64(PyObject* self, PyObject* args){
//ARGS: (double tempK, double pressureAtm, double conc, int wavenumRes, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, char* runID)
	double tempK;
	double pressureAtm;
	double conc;
	int wavenumRes;
	double startWavenum;
	double endWavenum;
	char* gaasDir;
	char* moleculeID;
	char* runID;
	
	
	if (!PyArg_ParseTuple(args, "dddiddsss", &tempK, &pressureAtm, &conc, &wavenumRes, &startWavenum, &endWavenum, &gaasDir, &moleculeID, &runID)){
		return NULL;
	}
		
	double* spectrumTarget = (double*)malloc(wavenumRes*sizeof(double));//new float [wavenumRes];
	double* wavenumsTarget = (double*)malloc(wavenumRes*sizeof(double));
	
	runSimDouble(tempK, pressureAtm, conc, spectrumTarget, wavenumsTarget, wavenumRes, startWavenum, endWavenum, gaasDir, moleculeID, runID);
	
	PyObject* spectrumList = PyList_New(wavenumRes);
	PyObject* wavenumsList = PyList_New(wavenumRes);
	PyObject* tempVal;
	
	for(int i = 0; i < wavenumRes; i++){
		//add results to py lists for output
		tempVal = Py_BuildValue("d",spectrumTarget[i]);
		PyList_SetItem(spectrumList, i, tempVal);
		
		tempVal = Py_BuildValue("d",wavenumsTarget[i]);
		PyList_SetItem(wavenumsList, i, tempVal);

	}
	
	PyObject* output = PyTuple_New(2);
	if ( !(PyTuple_SetItem(output, 0, spectrumList) == 0))
		printf("Set output tuple 0 failed...");
	if ( !(PyTuple_SetItem(output, 1, wavenumsList) == 0))
		printf("Set output tuple 1 failed...");
		
	free(spectrumTarget);
	free(wavenumsTarget);
	
	return output;
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "runSimF32", runSimF32, METH_VARARGS, "est" },
    { "runSimF64", runSimF64, METH_VARARGS, "est" },
    { NULL, NULL, 0, NULL }
};
// Our Module Definition struct
static struct PyModuleDef gaasAPI = {
    PyModuleDef_HEAD_INIT,
    "gaasAPI",
    "Test Module",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_gaasAPI(void)
{
    return PyModule_Create(&gaasAPI);
    }
