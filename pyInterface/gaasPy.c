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
 
#include "gaasCWrapper.h"
#include <Python.h>
#include <stdio.h>

static PyObject* sim_voigt(PyObject* self, PyObject* args){
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

	simVoigt(tempK, pressureAtm, conc, spectrumTarget, wavenumsTarget, wavenumStep, startWavenum, endWavenum, gaasDir, moleculeID, isoNum, runID);
	
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

static PyObject* sim_htp(PyObject* self, PyObject* args){
	//	list of tuples  		   float        float            float               float                float            
//ARGS: featureDataHTP* features, float tempK, float molarMass, double wavenumStep, double startWavenum, double endWavenum
//feature struct tuple:
	printf("-1");

	PyObject* features_in; //list of tuples 
	double temp_in;
	double molarMass_in;
	double wavenumStep_in;
	double startWavenum_in;
	double endWavenum_in;
	
	if (!PyArg_ParseTuple(args, "Oddddd", &features_in, &temp_in, &molarMass_in, &wavenumStep_in, &startWavenum_in, &endWavenum_in)){
		printf("Failed to parse args\n");
		return NULL;
	}
	printf("0");
	if(!PyList_Check(features_in)){
		printf("Error: must pass a list as the first argument to gaas.sim_htp");
		return NULL;
	}
	//parse features
	int numFeatures = PyList_Size(features_in);

	struct featureDataHTP* features = (struct featureDataHTP*)malloc(sizeof(struct featureDataHTP)*numFeatures);
	printf("1");
	for(int i = 0; i < numFeatures; i++){
		PyObject* currTuple = PyList_GetItem(features_in,i);
		if(!PyTuple_Check(currTuple)){
			printf("Error gaas.sim_htp: features list must contain tuples\n");
			return NULL;
		}
		PyObject* currTupleItem;
		currTupleItem = PyTuple_GetItem(currTuple, 0);
		if(!currTupleItem){
			printf("Error gaas.sim_htp: Invalid tuple item at location 0");
			return NULL;
		}
		features[i].linecenter = PyFloat_AsDouble(currTupleItem);
		
		currTupleItem = PyTuple_GetItem(currTuple, 1);
		if(!currTupleItem){
			printf("Error gaas.sim_htp: Invalid tuple item at location 1");
			return NULL;
		}
		features[i].Gam0 = (float)PyFloat_AsDouble(currTupleItem);
		
		currTupleItem = PyTuple_GetItem(currTuple, 2);
		if(!currTupleItem){
			printf("Error gaas.sim_htp: Invalid tuple item at location 2");
			return NULL;
		}
		features[i].Gam2 = (float)PyFloat_AsDouble(currTupleItem);
		
		currTupleItem = PyTuple_GetItem(currTuple, 3);
		if(!currTupleItem){
			printf("Error gaas.sim_htp: Invalid tuple item at location 3");
			return NULL;
		}
		features[i].Delta0 = (float)PyFloat_AsDouble(currTupleItem);
		
		currTupleItem = PyTuple_GetItem(currTuple, 4);
		if(!currTupleItem){
			printf("Error gaas.sim_htp: Invalid tuple item at location 4");
			return NULL;
		}
		features[i].Delta2 = (float)PyFloat_AsDouble(currTupleItem);
		
		currTupleItem = PyTuple_GetItem(currTuple, 5);
		if(!currTupleItem){
			printf("Error gaas.sim_htp: Invalid tuple item at location 5");
			return NULL;
		}
		features[i].anuVC = (float)PyFloat_AsDouble(currTupleItem);
		
		currTupleItem = PyTuple_GetItem(currTuple, 6);
		if(!currTupleItem){
			printf("Error gaas.sim_htp: Invalid tuple item at location 6");
			return NULL;
		}
		features[i].eta = (float)PyFloat_AsDouble(currTupleItem);
		
		currTupleItem = PyTuple_GetItem(currTuple, 7);
		if(!currTupleItem){
			printf("Error gaas.sim_htp: Invalid tuple item at location 7");
			return NULL;
		}
		features[i].lineIntensity = (float)PyFloat_AsDouble(currTupleItem);
	}

	int numWavenums = (int)((endWavenum_in-startWavenum_in)/wavenumStep_in);
	float* spectrumTarget = (float*)malloc(numWavenums*sizeof(float));//new float [wavenumRes];
	double* wavenumsTarget = (double*)malloc(numWavenums*sizeof(double));
	
	printf("successfully parsed");
	free(spectrumTarget);
	free(wavenumsTarget);
	free(features);
	return features_in;
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "sim_voigt", sim_voigt, METH_VARARGS, "est" },
    { "sim_htp", sim_htp, METH_VARARGS, "est" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef gaasAPI = {
    PyModuleDef_HEAD_INIT,
    "gaasAPI",
    "GPU Accelerated Absorption Simulation",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_gaasAPI(void)
{
    return PyModule_Create(&gaasAPI);
    }
