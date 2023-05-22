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
import matplotlib.pyplot as plt
import time
import tipsDB.genTIPSFile as gt
from os import listdir
import numpy as np
import pyOpenclVersion.opencl.GAASOpenCL as gaasApi
import math

# this tells GAAS how many wavenumbers to add to the extremities of the spectrum - the spectrum is then pared back to the 
# original size. This improves the accurracy of the simulation at the min and max wavenumbers of the spectrum
WAVENUMBUFFER = 50

g_api = gaasApi.Gaas_OCL_API()

def init(startWavenum, endWavenum, moleculeID, isotopologueID, gaasDirectory, ParDirectory, id, loadFromHITRAN=False):
    """
    Generates absorption database related files in a compact binary format which allows GAAS binary to quickly
    load absorption parameters when running multiple simulations. Also loads a new TIPS file if there isnt one in the
    current directory
    :param startWavenum: First Wavenumber to simulate, any features with linecenter less than this are ignored
    :param endWavenum: Last wavenumber to simulate, any features with linecenter greater than this are ignored
    :param moleculeID: HITRAN id of molecule to simulate ex: 'H2O', 'CO2' etc.
    :param gaasDirectory: Directory path to store Gaas program data
    :param ParDirectory: directory containing HITRAN Par files
    :param id: id to refer to this initialization when running simulation
    :param loadFromHITRAN: specify True if you want to download par files from the Hitran server
    :param isotopologueID: default is 1, specifies the isotopologue (integer) of the molecule being used, matches the indexing of
    isotopologues on HITRAN
    :return: void
    """

    save_absorption_db(moleculeID, isotopologueID, gaasDirectory+moleculeID+"_iso_"+str(isotopologueID)+"_"+id, max(
        startWavenum-WAVENUMBUFFER, 0), max(endWavenum+WAVENUMBUFFER, 0), ParDirectory, loadFromHITRAN=loadFromHITRAN)

    # check for TIPS file in gaasDirectory.
    tipsFilename = moleculeID + "_iso_" + str(isotopologueID) + "_tips.csv"
    shouldGenTips = True

    for f in listdir(gaasDirectory):
        if f == tipsFilename:
            shouldGenTips = False
    if shouldGenTips:
        print("generating TIPS file")
        gt.generateTIPSFile(moleculeID, isotopologueID, gaasDirectory)

def getHITRANMolecules():
    return ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3',
                            'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2', 'NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3']

def gen_abs_db(moleculeID, isotopologueNum, minWavenum, maxWavenum, parDirectory, strengthCutoff=0, loadFromHITRAN=False):
    """
    Saves absorption database in a compact format which can be read by gaas executable
    :param moleculeID:  string of molecule of interest ex. 'H2O'
    :param minWavenum: min wavenum
    :param maxWavenum: max wavenum
    :param parLocation: location of .par files
    :param strengthCutoff: minimum reference linestrength to include
    :param loadFromHITRAN: True= dowload data from HITRAN or False= Use current HITRAN database file in hapiLocation,
    :return: void
    """
    minWavenumAdj = max(minWavenum-WAVENUMBUFFER, 0)
    maxWavenumAdj = max(maxWavenum+WAVENUMBUFFER, 0)

    # This may not be necessary on every system
    ssl._create_default_https_context = ssl._create_unverified_context
    hapi.db_begin(parDirectory)
    if loadFromHITRAN:
        HITRAN_molecules = getHITRANMolecules() 
        
        molecule_number = (HITRAN_molecules.index(moleculeID)) + 1
        hapi.fetch(moleculeID, molecule_number,
                   isotopologueNum, minWavenumAdj-1, maxWavenumAdj+1,ParameterGroups=['160-char'])

    hapi.describeTable(moleculeID)
    nu, n_air, gamma_air, gamma_self, sw, elower, deltaAir = hapi.getColumns(
        moleculeID, ['nu', 'n_air', 'gamma_air', 'gamma_self', 'sw', 'elower', 'delta_air'])

    out = np.empty_like(nu,dtype=g_api.getVoigtDBStructDatatype())
    
    for i in range(len(nu)):
        # t = out[i]
        # if (sw[i] >= strengthCutoff and nu[i] >= minWavenumAdj and nu[i] <= maxWavenumAdj):
        out[i]['transWavenum'] = nu[i]
        out[i]['nAir'] = n_air[i]
        out[i]['gammaAir'] = gamma_air[i]
        out[i]['gammaSelf'] = gamma_self[i]
        out[i]['refStrength'] = sw[i]
        out[i]['ePrimePrime'] = elower[i]
        out[i]['deltaAir'] = deltaAir[i]

    #remove existing cache item 
    for i in range(len(abs_db_cache)-1,-1,-1):
        if(abs_db_cache[i][0]==moleculeID and abs_db_cache[i][1]==isotopologueNum):
            abs_db_cache.remove(i) #this is safe since we are iterating from end to beginning
    
    abs_db_cache.append((moleculeID,isotopologueNum,minWavenum,maxWavenum,out))
    return out

abs_db_cache = [] #list of all loaded absorption databases in this format [(molID,iso_ID,minWvn,maxWvn,abs_db)]

def get_or_gen_abs_db(moleculeID, isoNum, minWavenum, maxWavenum, parDirectory, strengthCutoff=0, loadFromHITRAN=False):
    #see if cache contains DB via brute force
    for db_cache in abs_db_cache:
        if(db_cache[0]==moleculeID and db_cache[1]==isoNum):
            if(minWavenum>=db_cache[2] and maxWavenum<=db_cache[3]):
                return db_cache[4]
            
    return gen_abs_db(moleculeID, isoNum, minWavenum, maxWavenum, parDirectory, strengthCutoff=strengthCutoff, loadFromHITRAN=loadFromHITRAN)

def clear_abs_db_cache():
    abs_db_cache = []

def get_tips_calc(moleculeID, isotopologueNum):
    tipskey = moleculeID+str(isotopologueNum)
    if(tipskey in get_tips_calc.cache):
        return get_tips_calc.cache[tipskey]
    else:
        get_tips_calc.cache[tipskey] = gt.TIPsCalculator(moleculeID,isotopologueNum)
        return get_tips_calc.cache[tipskey]

get_tips_calc.cache = {}

def simVoigt(tempK, pressureAtm, conc,  wavenumStep, startWavenum, endWavenum, moleculeID, isotopologueID, absDB, tipsCalc):
    """
    :param tempK:
    :param pressureAtm:
    :param conc: molar concentration - pathlength is assumed to be 1cm, scale the spectra after to account for larger pathlength.
    :param wavenumStep: wavenumbers between each simulation sample, higher wavenumStep = faster
    :param startWavenum: first wavenumber to simulate
    :param endWavenum: last wavenumber to simulate
    :param gaasDir: gaas directory specified in init
    :param moleculeID: HITRAN Molecule ID
    :param runID: Use multiple run ids if you want to run different simulations at the same time (ie with different molecules or wavenumber ranges)
    :return: (spectrum : list, wavenums : list)
    """
    startWavenumAdj = max(startWavenum-WAVENUMBUFFER, 0)
    endWavenumAdj = max(endWavenum+WAVENUMBUFFER, 0)

    wvn, a_coefs = g_api.voigtSim(
                                absDB,
                                tempK,
                                pressureAtm,
                                conc,
                                tipsCalc.getQ(273),
                                tipsCalc.getQ(tempK),
                                startWavenumAdj,
                                wavenumStep,
                                endWavenumAdj,
                                g_api.molMassMap[moleculeID+str(isotopologueID)],
                                g_api.isoAbundanceMap[moleculeID+str(isotopologueID)])
    
    buff = int(WAVENUMBUFFER/wavenumStep)
    # return (wvn[buff:(len(wvn)-buff+1)], a_coefs[buff:(len(a_coefs)-buff+1)])
    return (wvn,a_coefs)

def db_begin_gaas(parDirectory):
    gaas_par_directory = parDirectory
gaas_par_directory = None #only used for HAPI emulation

def absorptionCoefficient_Voigt_gaas(Components=None,SourceTables=None,partitionFunction=None,
                                Environment=None,OmegaRange=None,OmegaStep=None,OmegaWing=None,
                                IntensityThreshold=0,
                                OmegaWingHW=50,
                                GammaL='gamma_air', HITRAN_units=True, LineShift=True,
                                File=None, Format=None, OmegaGrid=None,
                                WavenumberRange=None,WavenumberStep=None,WavenumberWing=None,
                                WavenumberWingHW=None,WavenumberGrid=None,
                                Diluent={},EnvDependences=None,LineMixingRosen=False):
    """
    Wrapper to emulate HAPI behavior:

    INPUT PARAMETERS: 
        Components:  list of tuples [(M,I,D)], where
                        M - HITRAN molecule number,
                        I - HITRAN isotopologue number,
                        D - relative abundance (optional)
        SourceTables:  list of tables from which to calculate cross-section   (optional) NOT USED
        partitionFunction:  pointer to partition function (default is PYTIPS) (optional) NOT USED
        Environment:  dictionary containing thermodynamic parameters.
                        'p' - pressure in atmospheres,
                        'T' - temperature in Kelvin
                        Default={'p':1.,'T':296.}
        WavenumberRange:  wavenumber range to consider.
        WavenumberStep:   wavenumber step to consider. 
        WavenumberWing:   absolute wing for calculating a lineshape (in cm-1) 
        WavenumberWingHW:  relative wing for calculating a lineshape (in halfwidths)
        IntensityThreshold:  threshold for intensities
        GammaL:  specifies broadening parameter ('gamma_air' or 'gamma_self')
        HITRAN_units:  use cm2/molecule (True) or cm-1 (False) for absorption coefficient
        File:   write output to file (if specified)
        Format:  c-format of file output (accounts for significant digits in WavenumberStep)
        LineMixingRosen: include 1st order line mixing to calculation
    OUTPUT PARAMETERS: 
        Wavenum: wavenumber grid with respect to parameters WavenumberRange and WavenumberStep
        Xsect: absorption coefficient calculated on the grid
    ---
    DESCRIPTION:
        Calculate absorption coefficient using HT profile.
        Absorption coefficient is calculated at arbitrary temperature and pressure.
        User can vary a wide range of parameters to control a process of calculation.
        The choise of these parameters depends on properties of a particular linelist.
        Default values are a sort of guess which gives a decent precision (on average) 
        for a reasonable amount of cpu time. To increase calculation accuracy,
        user should use a trial and error method.
    ---
    EXAMPLE OF USAGE:
        nu,coef = absorptionCoefficient_HT(((2,1),),'co2',WavenumberStep=0.01,
                                              HITRAN_units=False,GammaL='gamma_self')
    """

    tempK = Environment["T"]
    pressureAtm = Environment["p"]

    # map molecule ID
    HITRAN_molecules = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3',
                    'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2',
                    'NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3']
    
    if(WavenumberGrid is None):
        if(WavenumberStep is not None):
            wavenumStep = WavenumberStep
            startWavenum = WavenumberRange[0]
            endWavenum = WavenumberRange[1]
    else:
        wavenumStep = WavenumberGrid[1]-WavenumberGrid[0]
        startWavenum = WavenumberGrid[0]
        endWavenum = WavenumberGrid[-1]

    nus = None
    coefs_summed = None
    for comp in Components:
        molecule_ID = HITRAN_molecules[comp[0]]
        iso_ID = comp[1]
        conc = comp[2]
        if(Diluent != {}):
            conc = Diluent["self"]
        tipsCalc = get_tips_calc(molecule_ID,iso_ID)
        if(gaas_par_directory is None):
            print("Error GAAS.absorptionCoefficient_Voigt_gaas - you need to call db_begin_gaas before using absorptionCoefficient_Voigt_gaas")
            exit(-1)

        absDB = get_or_gen_abs_db(molecule_ID,iso_ID,startWavenum,endWavenum,gaas_par_directory)
        (nus, coefs) = simVoigt(tempK, pressureAtm, conc,  wavenumStep, startWavenum, endWavenum, molecule_ID, iso_ID, absDB, tipsCalc)
        if(coefs_summed is None):
            coefs_summed = coefs
        else:
            coefs_summed = coefs_summed+coefs

    return nus, coefs_summed



class HTPFeatureData:
    #used to pass a list of feature data objects to simHTP
    
    def __init__(self, linecenter: float, Gam0: float, Gam2: float, Delta0: float, Delta2: float, anuVC: float, eta: float, lineIntensity: float) -> None:
        self.dataList = [linecenter,Gam0,Gam2,Delta0,Delta2,anuVC,eta,lineIntensity]

    def getDataTuple(self):
        return (self.dataList[0],self.dataList[1],self.dataList[2],self.dataList[3],self.dataList[4],self.dataList[5],self.dataList[6],self.dataList[7])

def simHTP_legacy(features, tempK, molarMass, wavenumStep, startWavenum, endWavenum):
    """
    Runs HTP simulation using GAAS, simulates each feature in features to produce an absorbance spectrum and wavenumber array
    :param features: list of HTPFeatureData objects
    :param tempK: temperature in Kelvin
    :param molarMass: molar mass of absorber
    :param wavenumStep: wavenumber resolution
    :param startWavenum: wavenumber range start
    :param endWavenum: wavenumber range end
    :return: (spectrum, wavenums)
    """
    htp_dbtype = g_api.getHTPDBStructDatatype()
    dbArray = np.empty(len(features),dtype=htp_dbtype)
# [('transWavenum','<f8'),
#     ('Gam0','<f8'),
#     ('Gam2','<f8'),
#     ('Delta0','<f8'),
#     ('Delta2','<f8'),
#     ('anuVC','<f8'),
#     ('eta','<f8'),
#     ('lineIntensity','<f8')]


# [linecenter,Gam0,Gam2,Delta0,Delta2,anuVC,eta,lineIntensity]
    for i in range(len(features)):
        dbArray[i]['transWavenum'] = features[i].dataList[0]
        dbArray[i]['Gam0'] =         features[i].dataList[1]
        dbArray[i]['Gam2'] =         features[i].dataList[2]
        dbArray[i]['Delta0'] =       features[i].dataList[3]
        dbArray[i]['Delta2'] =       features[i].dataList[4]
        dbArray[i]['anuVC'] =        features[i].dataList[5]
        dbArray[i]['eta'] =          features[i].dataList[6]
        dbArray[i]['lineIntensity'] =features[i].dataList[7]
    
    return g_api.HTPSim(dbArray,tempK,startWavenum,wavenumStep,endWavenum,molarMass)

def simHTP(tempK, pressureAtm, conc,  wavenumStep, startWavenum, endWavenum, moleculeID, isotopologueID, absDB, tipsCalc):
    return

def gen_abs_db_ht(moleculeID, isotopologueNum, minWavenum, maxWavenum, parDirectory, strengthCutoff=0, loadFromHITRAN=False):
    """
    Generates absorption database for hartmann tran profiles
    :param moleculeID:  string of molecule of interest ex. 'H2O'
    :param minWavenum: min wavenum
    :param maxWavenum: max wavenum
    :param parLocation: location of .par files
    :param strengthCutoff: minimum reference linestrength to include
    :param loadFromHITRAN: True= dowload data from HITRAN or False= Use current HITRAN database file in hapiLocation,
    :return: void
    """
    minWavenumAdj = max(minWavenum-WAVENUMBUFFER, 0)
    maxWavenumAdj = max(maxWavenum+WAVENUMBUFFER, 0)

    # This may not be necessary on every system
    ssl._create_default_https_context = ssl._create_unverified_context
    hapi.db_begin(parDirectory)
    if loadFromHITRAN:
        HITRAN_molecules = getHITRANMolecules() 
        molecule_number = (HITRAN_molecules.index(moleculeID)) + 1
        hapi.fetch(moleculeID, molecule_number,
                   isotopologueNum, minWavenumAdj-1, maxWavenumAdj+1,ParameterGroups=['160-char', ])

    hapi.describeTable(moleculeID)
    nu, n_air, gamma_air, gamma_self, sw, elower, deltaAir = hapi.getColumns(
        moleculeID, ['nu', 'n_air', 'gamma_air', 'gamma_self', 'sw', 'elower', 'delta_air'])

    out = np.empty_like(nu,dtype=g_api.getVoigtDBStructDatatype())
    
    for i in range(len(nu)):
        # t = out[i]
        # if (sw[i] >= strengthCutoff and nu[i] >= minWavenumAdj and nu[i] <= maxWavenumAdj):
        out[i]['transWavenum'] = nu[i]
        out[i]['nAir'] = n_air[i]
        out[i]['gammaAir'] = gamma_air[i]
        out[i]['gammaSelf'] = gamma_self[i]
        out[i]['refStrength'] = sw[i]
        out[i]['ePrimePrime'] = elower[i]
        out[i]['deltaAir'] = deltaAir[i]

    # print("saving ", len(absParamData)/7, " lines.")
    return out

def _gammaD_htp(molarMass, tempK, transWvn):
    cMassMol = 1.66053873e-27
    cBolts = 1.380648813E-16
    cc = 2.99792458e10
    LOG2 = 0.69314718056
    m = molarMass * cMassMol * 1000
    return math.sqrt(2 * cBolts * tempK * LOG2 / m / (cc * cc)) * transWvn