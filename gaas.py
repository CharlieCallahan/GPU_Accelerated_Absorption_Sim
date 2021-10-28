import hapi
import ssl
import struct
import gaasAPI
import matplotlib.pyplot as plt
import time

def gaasInit(startWavenum, endWavenum, moleculeID, gaasDirectory, HITRANParDirectory, id, loadFromHITRAN=False):
    """
    :param startWavenum: First Wavenumber to simulate, any features with linecenter less than this are ignored
    :param endWavenum: Last wavenumber to simulate, any features with linecenter greater than this are ignored
    :param moleculeID: HITRAN id of molecule to simulate ex: 'H2O', 'CO2' etc.
    :param gaasDirectory: Directory path to store Gaas program data
    :param HITRANParDirectory: directory containing HITRAN Par files
    :param id: id to refer to this initialization when running simulation
    :param loadFromHITRAN: specify True if you want to download par files from Hitran server
    :return: void
    """
    saveAbsorptionDB(moleculeID, gaasDirectory+moleculeID+id, startWavenum, endWavenum,HITRANParDirectory, loadFromHITRAN=loadFromHITRAN)

def saveAbsorptionDB(moleculeID, filename, minWavenum, maxWavenum, hapiLocation ,strengthCutoff=0, loadFromHITRAN=False):
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

    ssl._create_default_https_context = ssl._create_unverified_context
    hapi.db_begin(hapiLocation)

    if loadFromHITRAN:
        HITRAN_molecules = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3',
                            'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl'
            , 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2',
                            'NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3']
        molecule_number = (HITRAN_molecules.index(moleculeID)) + 1
        hapi.fetch(moleculeID, molecule_number, 1, minWavenum-1, maxWavenum+1)

    hapi.describeTable(moleculeID)
    nu,n_air,gamma_air,gamma_self,sw,elower,deltaAir = hapi.getColumns(moleculeID,['nu','n_air','gamma_air','gamma_self','sw','elower','delta_air'])
    absParamData = []
    for i in range(len(nu)):
        if (sw[i] >= strengthCutoff and nu[i] >= minWavenum and nu[i] <= maxWavenum):
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

def gaasRunF32(tempK, pressureAtm, conc,  wavenumRes, startWavenum, endWavenum, gaasDir, moleculeID, runID):
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
    output = gaasAPI.runSimF32(tempK,pressureAtm,conc,wavenumRes,startWavenum,endWavenum,gaasDir,moleculeID,runID)
    return output

def gaasRunF64(tempK, pressureAtm, conc,  wavenumRes, startWavenum, endWavenum, gaasDir, moleculeID, runID):
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
    output = gaasAPI.runSimF64(tempK,pressureAtm,conc,wavenumRes,startWavenum,endWavenum,gaasDir,moleculeID,runID)
    return output

def runHAPI(tempK, pressureAtm, conc,  wavenumRes, startWavenum, endWavenum, moleculeID):
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
    hapi.db_begin('HTData')
    HITRAN_molecules = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3',
                        'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl'
        , 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2',
                        'NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3']
    molecule_number = (HITRAN_molecules.index(moleculeID)) + 1
    wavenumStep = (endWavenum - startWavenum)/wavenumRes
    t1 = time.time()
    nus, coefs = hapi.absorptionCoefficient_Voigt(Components=[(molecule_number, 1, conc)],
                                                  SourceTables=moleculeID,
                                                  Environment={'p': pressureAtm, 'T': tempK},
                                                  Diluent={'self': conc,
                                                           'air': 1 - conc},
                                                  WavenumberStep=wavenumStep, HITRAN_units=False)
    t2 = time.time()
    print("HAPI Time elapsed: ", (t2 - t1))
    return (nus,coefs)

#gaasInit(800,900,'H2O',"/home/gputestbed/Desktop/WMS_Processing_V2_ARPAE/gaas/",'HTData',"h2oTest",loadFromHITRAN=False)
# for i in range(300,1000,100):
# print("running HAPI")
# outHapi = runHAPI(300,1,.1,(1500-1000)*1000,1000,1500, "H2O")
# plt.plot(outHapi[0],outHapi[1])
out = gaasRunF64(4100,1,.1,(900-800)*500,800,900,"/home/gputestbed/Desktop/WMS_Processing_V2_ARPAE/gaas/","H2O","h2oTest")
plt.plot(out[1],out[0])
plt.show()