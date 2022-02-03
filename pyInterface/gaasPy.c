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
 
#include <Python.h>
#include "gaasCWrapper.h"
#include <stdio.h>

static PyObject* runSimF32(PyObject* self, PyObject* args){
//ARGS: (double tempK, double pressureAtm, double conc, double wavenumStep, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, char* runID)
	double tempK;
	double pressureAtm;
	double conc;
	double wavenumStep;
	double startWavenum;
	double endWavenum;
	char* gaasDir;
	char* moleculeID;
	int isoNum;
	char* runID;
	
	
	if (!PyArg_ParseTuple(args, "ddddddssis", &tempK, &pressureAtm, &conc, &wavenumStep, &startWavenum, &endWavenum, &gaasDir, &moleculeID, &isoNum, &runID)){
		return NULL;
	}
	int numWavenums = (int)((endWavenum-startWavenum)/wavenumStep);
	float* spectrumTarget = (float*)malloc(numWavenums*sizeof(float));//new float [wavenumRes];
	double* wavenumsTarget = (double*)malloc(numWavenums*sizeof(double));

	runSimFloat(tempK, pressureAtm, conc, spectrumTarget, wavenumsTarget, wavenumStep, startWavenum, endWavenum, gaasDir, moleculeID, isoNum, runID);
	
	PyObject* spectrumList = PyList_New(numWavenums);
	PyObject* wavenumsList = PyList_New(numWavenums);
	PyObject* tempVal;

	for(int i = 0; i < numWavenums; i++){
	//add results to py lists for output
	tempVal = Py_BuildValue("f",spectrumTarget[i]);
	PyList_SetItem(spectrumList, i, tempVal);
	
	tempVal = Py_BuildValue("d",wavenumsTarget[i]);
	PyList_SetItem(wavenumsList, i, tempVal);
	}
	
	PyObject* output = PyTuple_New(2);
	if ( !(PyTuple_SetItem(output, 0, wavenumsList) == 0))
		printf("Set output tuple 0 failed...");
	if ( !(PyTuple_SetItem(output, 1, spectrumList) == 0))
		printf("Set output tuple 1 failed...");
		
	free(spectrumTarget);
	free(wavenumsTarget);
	return output;
}

//runs simulation with double point precision, requires cuda architecture > 6.0
static PyObject* runSimF64(PyObject* self, PyObject* args){
//ARGS: (double tempK, double pressureAtm, double conc, double wavenumStep, double startWavenum, double endWavenum, char* gaasDir, char* moleculeID, char* runID)
	double tempK;
	double pressureAtm;
	double conc;
	double wavenumStep; //number of wavenumbers to put into wavenum grid
	double startWavenum;
	double endWavenum;
	char* gaasDir;
	char* moleculeID;
	int isoNum;
	char* runID;
	
	
	if (!PyArg_ParseTuple(args, "ddddddssis", &tempK, &pressureAtm, &conc, &wavenumStep, &startWavenum, &endWavenum, &gaasDir, &moleculeID, &isoNum, &runID)){
		return NULL;
	}
	int numWavenums = (int)((endWavenum-startWavenum)/wavenumStep);
	double* spectrumTarget = (double*)malloc(numWavenums*sizeof(double));//new float [wavenumRes];
	double* wavenumsTarget = (double*)malloc(numWavenums*sizeof(double));
	
	runSimDouble(tempK, pressureAtm, conc, spectrumTarget, wavenumsTarget, wavenumStep, startWavenum, endWavenum, gaasDir, moleculeID, isoNum, runID);
	
	PyObject* spectrumList = PyList_New(numWavenums);
	PyObject* wavenumsList = PyList_New(numWavenums);
	PyObject* tempVal;
	
	for(int i = 0; i < numWavenums; i++){
		//add results to py lists for output
		tempVal = Py_BuildValue("d",spectrumTarget[i]);
		PyList_SetItem(spectrumList, i, tempVal);
		
		tempVal = Py_BuildValue("d",wavenumsTarget[i]);
		PyList_SetItem(wavenumsList, i, tempVal);

	}
	
	PyObject* output = PyTuple_New(2);
	if ( !(PyTuple_SetItem(output, 0, wavenumsList) == 0))
		printf("Set output tuple 0 failed...");
	if ( !(PyTuple_SetItem(output, 1, spectrumList) == 0))
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
