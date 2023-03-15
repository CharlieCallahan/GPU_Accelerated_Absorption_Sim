import decimal
import numpy as np
import pickle
import csv
import os
# Copyright (c) 2021 Charlie Callahan

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

# MIT LICENSE
#
# COPYRIGHT (C) [2021] [ROBERT GAMACHE]
#
# PERMISSION IS HEREBY GRANTED, FREE OF CHARGE, TO ANY PERSON OBTAINING A COPY
# OF THIS SOFTWARE AND ASSOCIATED DOCUMENTATION FILES (THE "SOFTWARE"), TO DEAL
# IN THE SOFTWARE WITHOUT RESTRICTION, INCLUDING WITHOUT LIMITATION THE RIGHTS
# TO USE, COPY, MODIFY, MERGE, PUBLISH, DISTRIBUTE, SUBLICENSE, AND/OR SELL
# COPIES OF THE SOFTWARE, AND TO PERMIT PERSONS TO WHOM THE SOFTWARE IS
# FURNISHED TO DO SO, SUBJECT TO THE FOLLOWING CONDITIONS:
#
# THE ABOVE COPYRIGHT NOTICE AND THIS PERMISSION NOTICE SHALL BE INCLUDED IN ALL
# COPIES OR SUBSTANTIAL PORTIONS OF THE SOFTWARE.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
#  --  UPDATES  --
#    C    FOR UPDATES SEE SEE PUBLICATION 
#    GAMACHE ET AL., Total Internal Partition Sums for the HITRAN2020 database, JQSRT, ??, ??, 2021. 
#
#    THIS PROGRAM CALCULATES THE TOTAL INTERNAL
#    PARTITION SUM (TIPS) FOR A GIVEN MOLECULE,ISOTOPOLOGUE, AND
#    TEMPERATURE.  CURRENT LIMITATIONS ARE THE MOLECULAR SPECIES ON THE
#    HITRAN MOLECULAR DATABASE PLUS A FEW ADDITIONAL MOLECULES AND THE TEMPERATURE RANGE IS GENERALLY 1 - 5000 K.


molecules = [' ','H2O','CO2','O3','N2O','CO','CH4','O2',
'NO','SO2','NO2','NH3','HNO3','OH','HF','HCl',
'HBr','HI','ClO','OCS','H2CO','HOCl','N2','HCN',
'CH3Cl','H2O2','C2H2','C2H6','PH3','COF2','SF6','H2S',
'HCOOH','HO2','O','ClONO2','NO+','HOBr','C2H4','CH3OH',
'CH3Br','CH3CN','CF4','C4H2','HC3N','H2','CS','SO3','C2N2', 
'COCl2','SO','CH3F','GeH4','CS2','CH3I','NF3','C3H4','CH3']

niso = [0,9,13,18,5,9,4,6,3,4,2,2,2,3,2,4,4,2,2,6,3,2,3,3,2,1,3,3,1,
2,1,3,1,1,1,2,1,2,3,1,2,4,1,1,6,2,4,1,2,2,3,1,5,4,2,1,1,1]

# H2O
Tmax = [0, 5000.,5000.,5000.,5000.,5000.,5000.,6000.,6000.,6000.,
# CO2
5000.,5000.,3500.,3500.,3500.,3500.,5000.,3500.,5000.,5000.,3500.,5000.,5000.,
# O3
1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,1000.,
# N2O,                           CO
5000.,5000.,5000.,5000.,5000.,   9000.,9000.,9000.,9000.,9000.,9000.,9000.,9000.,9000.,
# CH4,                     O2,                                    NO,                  SO2
2500.,2500.,2500.,2500.,   7500.,7500.,7500.,7500.,7500.,7500.,   5000.,5000.,5000.,   5000.,5000.,5000.,5000.,
# NO2,         NH3,           HNO3,          OH,                  HF,            HCl
1000.,1000.,   6000.,6000.,   3500.,3500.,   9000.,5000.,5000.,   6000.,6000.,   6000.,6000.,6000.,6000.,
# HBr,                     HI,            ClO,           OCS,                                   H2CO
6000.,6000.,6000.,6000.,   6000.,6000.,   5000.,5000.,   5000.,5000.,5000.,5000.,5000.,5000.,   3500.,5000.,5000.,
# HOCl,        N2,                  HCN,                 CH3Cl,         H2O2,    C2H2, 
5000.,5000.,   9000.,9000.,9000.,   3500.,3500.,3500.,   5000.,5000.,   6000.,   5000.,5000.,5000.,   
# C2H6,              PH3,     COF2,          SF6,     H2S,                 HCOOH,   HO2,   O atom, ClONO2,
5000.,5000.,5000.,   4500.,   3500.,3500.,   5000.,   4000.,5000.,5000.,   5000.,   5000.,   0.,   5000.,5000.,   
#  NO+,  HOBr,          C2H4,                CH3OH,   CH3Br,         CH3CN,                     CF4
5000.,   5000.,5000.,   5000.,5000.,5000.,   3500.,   5000.,5000.,   5000.,5000.,5000.,5000.,   3010.,
# C4H2,  HC3N,                                  H2,            CS,                        SO3, 
5000.,   5000.,5000.,5000.,5000.,5000.,5000.,   6000.,6000.,   5000.,5000.,5000.,5000.,   3500.,
# C2N2,        COCl2,        SO,                 CH3F    GeH4                          
5000.,5000.,   5000.,5000.,  5000.,5000.,5000.,  5000.,  5000.,5000.,5000.,5000.,5000.,
# CS2                       CH3I,         NF3     C3H4,    CH3,  
  5000.,5000.,5000.,5000.,  5000.,5000.,  5000.,  5000.,   5000.]


def generateTIPSFile(moleculeName, isoNum, directory):
	#moleculeName = HITRAN molecule name string, ex: 'H2O'
	#isoNum = isotopologue number integer (hitran format: H_2^16 O = 1)
	#directory = directory to put resultant file

	mol = molecules.index(moleculeName)
	if not ((mol>0) and (mol<=57) and mol!=34): 
		print("Error: generateTIPSFile: molecule: ", moleculeName," not found.")
		exit(-1)
		
	iso = isoNum
	if not (iso>0 and iso<=niso[mol]):
		print("isotope # ",iso," out of range for molecule: ",moleculeName)
		print ('the range is',1,' to', niso[mol])
		exit(-1)
	  
	mol = str(mol)
	iso = str(iso)
	file = os.path.dirname(os.path.realpath(__file__)) + '/QTpy/'+mol+'_'+iso+'.QTpy'

	QTdict = {}
	with open(file, 'rb') as handle:
		QTdict = pickle.loads(handle.read())

	global_ID = 0
	for I in range(1,int(mol)):
		global_ID = global_ID + niso[I]
	global_ID = global_ID + int(iso)

	outputFilename = moleculeName + "_iso_" + str(isoNum) + "_tips.csv"

	if directory[-1] != "/":
		outputFilename = "/" + outputFilename
		
	with open(directory + outputFilename,'w') as csvFile:
		writer = csv.writer(csvFile, delimiter =',')
		for T in range(1,int(Tmax[global_ID])):
			#iterate over temps
			if(T==int(T)):
				key=str(int(T))
				QT = float(QTdict[key])
			else:
				key=str(int(T))
				Q1 = float(QTdict[key])
				key=str(int(T+1))
				Q2 = float(QTdict[key])
				QT = Q1+(Q2-Q1)*(T-int(T))
			writer.writerow([T,QT])

class TIPsCalculator:

	"""
	A class for calculating the TIPs for a given molecule/isotopologue
	"""
	def __init__(self, molID, isoID) -> None:
		mol = molecules.index(molID)
		if not ((mol>0) and (mol<=57) and mol!=34): 
			print("Error: generateTIPSFile: molecule: ", molID," not found.")
			exit(-1)
			
		iso = isoID
		if not (iso>0 and iso<=niso[mol]):
			print("isotope # ",iso," out of range for molecule: ",molID)
			print ('the range is',1,' to', niso[mol])
			exit(-1)
		
		mol = str(mol)
		iso = str(iso)
		file = os.path.dirname(os.path.realpath(__file__)) + '/QTpy/'+mol+'_'+iso+'.QTpy'

		QTdict = {}
		with open(file, 'rb') as handle:
			QTdict = pickle.loads(handle.read())

		global_ID = 0
		for I in range(1,int(mol)):
			global_ID = global_ID + niso[I]
		global_ID = global_ID + int(iso)

		#dict for temp to Q lookup
		self.Q = {}
		for T in range(1,int(Tmax[global_ID])):
			#iterate over temps
			if(T==int(T)):
				key=str(int(T))
				QT = float(QTdict[key])
			else:
				key=str(int(T))
				Q1 = float(QTdict[key])
				key=str(int(T+1))
				Q2 = float(QTdict[key])
				QT = Q1+(Q2-Q1)*(T-int(T))
			self.Q[T] = QT
	
	def getQ(self, temp):
		T0 = int(temp)
		T1 = T0+1
		Q0 = self.Q[T0]
		Q1 = self.Q[T1]
		return Q0 + (Q1-Q0)*(temp-T0)
