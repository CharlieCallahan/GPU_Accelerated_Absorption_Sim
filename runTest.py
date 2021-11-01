import gaas as gs
import os
import matplotlib.pyplot as plt

startWavenum = 2000
endWavenum = 4000
wavenumRes = 0.01 #wavenums per simulatuion step
mol = 'H2O' 
iso = 4 #isotopologue num
T = 300 #K
P = 1.0 #atm
conc = 0.1

cwd = os.path.dirname(os.path.realpath(__file__))

gs.gaasInit(startWavenum,endWavenum,mol,iso,cwd+"/gaasDir/",'HTData',"test",loadFromHITRAN=True)


nus_h,coefs_h = gs.runHAPI(T, P, conc, (endWavenum - startWavenum)/wavenumRes, startWavenum, endWavenum, mol, iso,'HTData')

nus, coefs = gs.gaasRunF32(T, P,conc,(endWavenum - startWavenum)/wavenumRes,startWavenum,endWavenum,cwd+"/gaasDir/",mol,iso,"test")

error = 0
total = 0 #area 

for i in range(len(coefs)-1):
    error+=abs(coefs_h[i]-coefs[i])
    total+=coefs_h[i]
    
print("Error: ",error/total*100,"%")

plt.plot(nus,coefs)
plt.plot(nus_h,coefs_h)
plt.legend(("HAPI","GAAS"))
plt.show()
