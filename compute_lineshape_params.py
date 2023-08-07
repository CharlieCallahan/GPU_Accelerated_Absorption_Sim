"""
This file has functions for computing HTP and Voigt lineshape parameters from a linelist
this is based on MATS from NIST
https://pages.nist.gov/MATS
"""
from gaas_ocl import HTPFeatureData
import tipsDB.genTIPSFile as gt
import math

T_ref = 296
LOG2 = math.log(2)

def getOptionalParam(key:str, paramDict:dict, defaultVal):
    if(key in paramDict.keys()):
        return paramDict[key]
    return defaultVal

"""
    Line params are specific to each line
    Line params should have the following keys:
    S_ref 
    E_pp
    nu_0 - Transition linecenter (cm-1)

    Envparams are the parameters used globally for the spectrum
    should have:
    T,
    molarMass,
    diluents: dict like {'CO2':0.1, 'N2':0.2, speciesID:concentration}

"""
def lineIntensity(lineParams:dict, envParams: dict, TIPSCalc:gt.TIPsCalculator):
    c2 = 1.4388028496642257; # cm*K
    Tref = getOptionalParam("T_ref",envParams,296)
    refStrength = lineParams["S_ref"]
    tempK = envParams["T"]
    pSumT = TIPSCalc.getQ(tempK)
    pSumTref = TIPSCalc.getQ(Tref)
    ePrimePrime = lineParams["E_pp"]
    transWavenum = lineParams["v_0"]
    return refStrength * pSumTref / pSumT * math.exp(-1 * c2 * ePrimePrime / tempK) / math.exp(-1 * c2 * ePrimePrime / Tref) * (1 - math.exp(-1 * c2 * transWavenum / tempK)) / (1 - math.exp(-1 * c2 * transWavenum / Tref))

def Gamma_d(lineParams:dict, envParams: dict):
    cMassMol = 1.66053873e-27
    cBolts = 1.380648813E-16 # erg/K
    cc = 2.99792458e10		 # cm/s
    molarMass = envParams["molarMass"]
    tempKelvin = envParams["T"]
    transWavenum = lineParams["v_0"]
    m = molarMass * cMassMol * 1000;
    return math.sqrt(2 * cBolts * tempKelvin * LOG2 / m / (cc * cc)) * transWavenum

def Gamma_0(lineParams:dict, envParams: dict):
    diluents = envParams["diluents"]
    gamma0 = 0
    P_ref = getOptionalParam("P_ref",envParams,1)
    T_ref = getOptionalParam("T_ref",envParams,296)
    T = getOptionalParam("T",envParams,296)
    P = getOptionalParam("P",envParams,1)
    for diluent in diluents.keys():
        gamma_0_diluent_param_id = "gamma0_"+diluent
        gamma_0_diluent = getOptionalParam(gamma_0_diluent_param_id,lineParams,0)

        n_gamma_0_diluent_param_id = "n_gamma0_"+diluent
        n_gamma_0_diluent = getOptionalParam(n_gamma_0_diluent_param_id,lineParams,0)

        diluent_concentration = diluents[diluent]
        gamma0 += diluent_concentration*gamma_0_diluent*P/P_ref*((T/T_ref)**(n_gamma_0_diluent))
    return gamma0

def Delta_0(lineParams:dict, envParams: dict):
    diluents = envParams["diluents"]
    delta_0 = 0
    P_ref = getOptionalParam("P_ref",envParams,1)
    T_ref = getOptionalParam("T_ref",envParams,296)
    T = getOptionalParam("T",envParams,296)
    P = getOptionalParam("P",envParams,1)

    for diluent in diluents.keys():
        delta_0_diluent_param_id = "delta0_"+diluent
        delta_0_diluent = getOptionalParam(delta_0_diluent_param_id,lineParams,0)

        n_delta_0_diluent_param_id = "n_delta0_"+diluent
        n_delta_0_diluent = getOptionalParam(n_delta_0_diluent_param_id,lineParams,0)
        diluent_concentration = diluents[diluent]

        delta_0 += diluent_concentration*(delta_0_diluent + n_delta_0_diluent*(T-T_ref))*P/P_ref
    return delta_0

def Gamma_2(lineParams:dict, envParams: dict):
    diluents = envParams["diluents"]
    gamma_2 = 0
    P_ref = getOptionalParam("P_ref",envParams,1)
    T_ref = getOptionalParam("T_ref",envParams,296)
    T = getOptionalParam("T",envParams,296)
    P = getOptionalParam("P",envParams,1)

    for diluent in diluents.keys():
        aw_diluent_id = "SD_gamma_"+diluent
        aw_dilent = getOptionalParam(aw_diluent_id,lineParams,0)

        gamma_0_diluent_id = "gamma0_"+diluent
        gamma_0_diluent = getOptionalParam(gamma_0_diluent_id,lineParams,0)

        n_gamma2_diluent_id = "n_gamma2_" + diluent
        n_gamma2_diluent = getOptionalParam(n_gamma2_diluent_id,lineParams,0)

        diluent_concentration = diluents[diluent]

        gamma_2 += diluent_concentration*aw_dilent*gamma_0_diluent*P/P_ref*((T_ref/T)**n_gamma2_diluent)

    return gamma_2

def Delta_2(lineParams:dict, envParams: dict):
    diluents = envParams["diluents"]
    delta_2 = 0
    P_ref = getOptionalParam("P_ref",envParams,1)
    T_ref = getOptionalParam("T_ref",envParams,296)
    T = getOptionalParam("T",envParams,296)
    P = getOptionalParam("P",envParams,1)

    for diluent in diluents.keys():
        as_diluent_id = "SD_shift_"+diluent
        as_dilent = getOptionalParam(as_diluent_id,lineParams,0)
        
        delta_0_diluent_id = "delta0_"+diluent
        delta_0 = getOptionalParam(delta_0_diluent_id,lineParams,0)

        n_delta_2_diluent_id = "n_delta2_"+diluent
        n_delta_2 = getOptionalParam(n_delta_2_diluent_id,lineParams,0)

        diluent_concentration = diluents[diluent]
        
        delta_2 += diluent_concentration*as_dilent*delta_0+n_delta_2*(T-T_ref)*P/P_ref

    return delta_2

