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


import hapi
import ssl
import struct
import gaasAPI
import matplotlib.pyplot as plt
import time
import tipsDB.genTIPSFile as gt
from os import listdir
import numpy as np

#adds this range of wavenumbers to the GAAS simulation then pares it back so that features at the extremes of the wavenumber
#are accurate 
WAVENUMBUFFER = 10

def gaasInit(startWavenum, endWavenum, moleculeID, isotopologueID, gaasDirectory, HITRANParDirectory, id, loadFromHITRAN=False):
    """
    Generates absorption database related files in a compact binary format which allows GAAS C library to quickly
    load absorption parameters when running multiple simulations. Also loads a new TIPS file if there isnt one in the
    current directory
    :param startWavenum: First Wavenumber to simulate, any features with linecenter less than this are ignored
    :param endWavenum: Last wavenumber to simulate, any features with linecenter greater than this are ignored
    :param moleculeID: HITRAN id of molecule to simulate ex: 'H2O', 'CO2' etc.
    :param gaasDirectory: Directory path to store Gaas program data
    :param HITRANParDirectory: directory containing HITRAN Par files
    :param id: id to refer to this initialization when running simulation
    :param loadFromHITRAN: specify True if you want to download par files from Hitran server
    :param isotopologueID: default is 1, specifies the isotopologue (integer) of the molecule being used, matches the indexing of
    isotopologues on HITRAN
    :return: void
    """
    
    saveAbsorptionDB(moleculeID, isotopologueID, gaasDirectory+moleculeID+"_iso_"+str(isotopologueID)+"_"+id, max(startWavenum-WAVENUMBUFFER,0),max(endWavenum+WAVENUMBUFFER,0), HITRANParDirectory, loadFromHITRAN=loadFromHITRAN)

    # check for TIPS file in gaasDirectory.
    tipsFilename = moleculeID + "_iso_" + str(isotopologueID) + "_tips.csv"
    shouldGenTips = True

    for f in listdir(gaasDirectory):
        if f == tipsFilename:
            shouldGenTips = False
    if shouldGenTips:
        print("generating TIPS file")
        gt.generateTIPSFile(moleculeID, isotopologueID, gaasDirectory)

def saveAbsorptionDB(moleculeID, isotopologueNum, filename, minWavenum, maxWavenum, hapiLocation, strengthCutoff=0, loadFromHITRAN=False):
    """
    Saves absorption database in a compact format which can be read by gaas executable
    :param moleculeID:  string of molecule of interest ex. 'H2O'
    :param filename: location to save gaas database
    :param minWavenum: min wavenum
    :param maxWavenum: max wavenum
    :param hapiLocation: location of HITRAN .par files
    :param strengthCutoff: minimum reference linestrength to include
    :param loadFromHITRAN: True= dowload data from HITRAN or False= Use current HITRAN database file in hapiLocation,
    :return: void
    """
    minWavenumAdj = max(minWavenum-WAVENUMBUFFER,0)
    maxWavenumAdj = max(maxWavenum+WAVENUMBUFFER,0)

    ssl._create_default_https_context = ssl._create_unverified_context #This may not be necessary on every system
    hapi.db_begin(hapiLocation)
    if loadFromHITRAN:
        HITRAN_molecules = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3',
                            'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2','NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3']
        molecule_number = (HITRAN_molecules.index(moleculeID)) + 1
        hapi.fetch(moleculeID, molecule_number,
                   isotopologueNum, minWavenumAdj-1, maxWavenumAdj+1)

    hapi.describeTable(moleculeID)
    nu, n_air,gamma_air,gamma_self,sw,elower,deltaAir = hapi.getColumns(moleculeID,['nu','n_air','gamma_air','gamma_self','sw','elower','delta_air'])
    absParamData = []

    for i in range(len(nu)):
        if (sw[i] >= strengthCutoff and nu[i] >= minWavenumAdj and nu[i] <= maxWavenumAdj):
            absParamData.append(nu[i])
            absParamData.append(n_air[i])
            absParamData.append(gamma_air[i])
            absParamData.append(gamma_self[i])
            absParamData.append(sw[i])
            absParamData.append(elower[i])
            absParamData.append(deltaAir[i])
    print("saving ", len(absParamData)/7, " lines.")

    filehandler = open(filename, 'wb')
    for i in range(len(absParamData)):
        filehandler.write(bytearray(struct.pack("<d", absParamData[i])))


def gaasRunF32(tempK, pressureAtm, conc,  wavenumRes, startWavenum, endWavenum, gaasDir, moleculeID, isotopologueID, runID):
    """
    runs simulation on GPU with 32 bit float precision
    suitable for older GPUs without support for atomicAdd(double *, double)  (Cuda architecture < 6.0 )
    Will also be faster for applications where double (64 bit) precision is not necessary.
    :param tempK:
    :param pressureAtm:
    :param conc:
    :param wavenumRes: number of wavenumbers to simulate over range, lower resolution = faster
    :param startWavenum: first wavenumber to simulate
    :param endWavenum: last wavenumber to simulate
    :param gaasDir: gaas directory specified in gaasInit
    :param moleculeID: HITRAN Molecule ID
    :param runID: Use multiple run ids if you want to run different simulations at the same time (ie with different molecules or wavenumber ranges)
    :return: (spectrum : list, wavenums : list)
    """
    # ARGS: (double tempK, double pressureAtm, double conc, int wavenumRes, double startWavenum, double endWavenum, char * gaasDir, char * moleculeID, char * runID)
    startWavenumAdj = max(startWavenum-WAVENUMBUFFER,0)
    endWavenumAdj = max(endWavenum+WAVENUMBUFFER,0)
    resAdj =int( wavenumRes*(endWavenumAdj-startWavenumAdj)/(endWavenum-startWavenum))

    nus,coefs = gaasAPI.runSimF32(tempK, pressureAtm,conc,int(resAdj),startWavenumAdj,endWavenumAdj,gaasDir,moleculeID,int(isotopologueID),runID)
    startPos = int(resAdj*WAVENUMBUFFER/(endWavenumAdj-startWavenumAdj))
    endPos = int(startPos+wavenumRes)
    return ( nus[startPos:endPos],coefs[startPos:endPos] )

def gaasRunF64(tempK, pressureAtm, conc,  wavenumRes, startWavenum, endWavenum, gaasDir, moleculeID, isotopologueID, runID):
    """
    runs simulation on GPU with 64 bit double precision
    requires Cuda architecture >= 6.0
    :param tempK:
    :param pressureAtm:
    :param conc:
    :param wavenumRes: number of wavenumbers to simulate over range, lower resolution = faster
    :param startWavenum: first wavenumber to simulate
    :param endWavenum: last wavenumber to simulate
    :param gaasDir: gaas directory specified in gaasInit
    :param moleculeID: HITRAN Molecule ID
    :param runID: Use multiple run ids if you want to run different simulations at the same time (ie with different molecules or wavenumber ranges)
    :return: (spectrum : list, wavenums : list)
    """
    # ARGS: (double tempK, double pressureAtm, double conc, int wavenumRes, double startWavenum, double endWavenum, char * gaasDir, char * moleculeID, char * runID)
    startWavenumAdj = max(startWavenum-WAVENUMBUFFER,0)
    endWavenumAdj = max(endWavenum+WAVENUMBUFFER,0)
    resAdj =int( wavenumRes*(endWavenumAdj-startWavenumAdj)/(endWavenum-startWavenum))

    nus,coefs = gaasAPI.runSimF64(tempK, pressureAtm,conc,int(resAdj),startWavenumAdj,endWavenumAdj,gaasDir,moleculeID,int(isotopologueID),runID)
    startPos = int(resAdj*WAVENUMBUFFER/(endWavenumAdj-startWavenumAdj))
    endPos = int(startPos+wavenumRes)
    return ( nus[startPos:endPos],coefs[startPos:endPos] )
    

def runHAPI(tempK, pressureAtm, conc,  wavenumRes, startWavenum, endWavenum, moleculeID, isotopologueID, hapiDB):
    """
    Runs simulation using HAPI python library, for performace testing baseline
    :param tempK:
    :param pressureAtm:
    :param conc:
    :param wavenumRes:
    :param startWavenum:
    :param endWavenum:
    :param moleculeID:
    :return: (spectrum, wavenums)
    """
    #hapi.db_begin(hapiDB)
    HITRAN_molecules = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3',
                        'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl'                        , 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2',
                        'NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3']
    molecule_number = (HITRAN_molecules.index(moleculeID)) + 1
    wavenumStep = (endWavenum - startWavenum)/wavenumRes
    
    nus, coefs = hapi.absorptionCoefficient_Voigt(Components=[(molecule_number, isotopologueID, conc)],
                                                  SourceTables=moleculeID,
                                                  Environment={
                                                      'p': pressureAtm, 'T': tempK},
                                                  Diluent={'self': conc,
                                                           'air': 1 - conc},
                                                  WavenumberStep=wavenumStep, HITRAN_units=False)

    return (nus, coefs)
