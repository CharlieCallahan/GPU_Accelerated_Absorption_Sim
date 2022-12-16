import hapi
def runHAPI(tempK, pressureAtm, conc,  wavenumStep, startWavenum, endWavenum, moleculeID, isotopologueID, hapiDB):
    """
    Runs simulation using HAPI python library, for performace testing baseline
    :param tempK:
    :param pressureAtm:
    :param conc:
    :param wavenumStep:
    :param startWavenum:
    :param endWavenum:
    :param moleculeID:
    :return: (spectrum, wavenums)
    """
    # hapi.db_begin(hapiDB)
    HITRAN_molecules = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3',
                        'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2',
                        'NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3']
    molecule_number = (HITRAN_molecules.index(moleculeID)) + 1

    nus, coefs = hapi.absorptionCoefficient_Voigt(Components=[(molecule_number, isotopologueID, conc)],
                                                  SourceTables=moleculeID,
                                                  Environment={
                                                      'p': pressureAtm, 'T': tempK},
                                                  Diluent={'self': conc,
                                                           'air': 1 - conc},
                                                  WavenumberRange=(
                                                      startWavenum, endWavenum),
                                                  WavenumberStep=wavenumStep, HITRAN_units=False)

    return (nus, coefs)

def runHAPI_HTP(tempK, pressureAtm, conc,  wavenumStep, startWavenum, endWavenum, moleculeID, isotopologueID, hapiDB):
    """
    Runs simulation using HAPI python library, for performace testing baseline
    :param tempK:
    :param pressureAtm:
    :param conc:
    :param wavenumStep:
    :param startWavenum:
    :param endWavenum:
    :param moleculeID:
    :return: (spectrum, wavenums)
    """
    # hapi.db_begin(hapiDB)
    HITRAN_molecules = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3',
                        'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2',
                        'NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3']
    molecule_number = (HITRAN_molecules.index(moleculeID)) + 1

    nus, coefs = hapi.absorptionCoefficient_HT(Components=[(molecule_number, isotopologueID, conc)],
                                                  SourceTables=moleculeID,
                                                  Environment={
                                                      'p': pressureAtm, 'T': tempK},
                                                  Diluent={'self': conc,
                                                           'air': 1 - conc},
                                                  WavenumberRange=(
                                                      startWavenum, endWavenum),
                                                  WavenumberStep=wavenumStep, HITRAN_units=False)
    return (nus, coefs)

