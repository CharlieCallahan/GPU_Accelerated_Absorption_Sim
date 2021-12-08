import gaas as gs
import os
import sys
import matplotlib.pyplot as plt

startWavenum = 2000
endWavenum = 3000
wavenumRes = 0.01 #wavenums per simulation step
mol = 'H2O' 
iso = 1 #isotopologue num
T = 300 #K
P = 1.0 #atm
conc = 0.1

cwd = os.path.dirname(os.path.realpath(__file__))
if sys.platform == 'win32' or sys.platform == 'win64':
    gaasDirPath = cwd + "\\gaasDir\\"
else:
    gaasDirPath = cwd + "/gaasDir/"
    
if (not os.path.isdir(gaasDirPath)):
    #need to make gaas directory
    os.mkdir(gaasDirPath)
    
gs.gaasInit(startWavenum,endWavenum,mol,iso,gaasDirPath,'HTData',"test",loadFromHITRAN=True)
                           #(tempK, pressureAtm, conc,  wavenumRes, startWavenum, endWavenum, moleculeID)
nus_h,coefs_h = gs.runHAPI(T, P, conc, (endWavenum - startWavenum)/wavenumRes, startWavenum, endWavenum, mol, iso,'HTData')

nus, coefs = gs.gaasRunF32(T, P,conc,(endWavenum - startWavenum)/wavenumRes,startWavenum,endWavenum,gaasDirPath,mol,iso,"test")


plt.plot(nus,coefs)
plt.plot(nus_h,coefs_h)
plt.legend(("GAAS","HAPI"))
plt.show()
