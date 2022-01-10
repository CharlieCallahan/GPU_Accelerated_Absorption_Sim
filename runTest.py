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

import gaas as gs
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time

startWavenum = 1300
endWavenum = 2250
wavenumRes = 0.001 #wavenums per simulation step
mol = 'H2O' #'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3'
iso = 1 #isotopologue num
T = 2500 #K
P = 1.0 #atm
conc = 0.05
#pathlength is assumed to be 100cm, if you want to use a different pathlenth, scale the absorbance by pl_cm/100

cwd = os.path.dirname(os.path.realpath(__file__))
if sys.platform == 'win32' or sys.platform == 'win64':
    gaasDirPath = cwd + "\\gaasDir\\"
else:
    gaasDirPath = cwd + "/gaasDir/"
    
if (not os.path.isdir(gaasDirPath)):
    #need to make gaas directory
    os.mkdir(gaasDirPath)
    
#gs.gaasInit(startWavenum,endWavenum,mol,iso,gaasDirPath,"C:\\Users\\Charlie\\Desktop\\HITEMP_DELETE_THIS\\db","test",loadFromHITRAN=False)
                           #(tempK, pressureAtm, conc,  wavenumRes, startWavenum, endWavenum, moleculeID)
#nus_h,coefs_h = gs.runHAPI(T, P, conc, (endWavenum - startWavenum)/wavenumRes, startWavenum, endWavenum, mol, iso,'HTData')
t1 = time.time()
nus, coefs = gs.gaasRunF32(T, P,conc,(endWavenum - startWavenum)/wavenumRes,startWavenum,endWavenum,gaasDirPath,mol,iso,"test")
print("sim time: ",time.time()-t1)
def getError(nus_h,coefs_h,nus_g,coefs_g):
    coefs_h_i = np.interp(nus_g,nus_h,coefs_h)
    err = 0
    sum = 0
    
    for i in range(len(coefs_h_i)):
        err+=abs(coefs_h_i[i]-coefs_g[i])
        sum+=coefs_h_i[i]
    return err/sum

#print("Simulation error = ",getError(nus_h,coefs_h,nus,coefs)*100,"%")
plt.plot(nus,coefs)
plt.show()
coefs_h_i = np.interp(nus,nus_h,coefs_h)
plt.plot(nus,coefs_h_i)
plt.plot(nus,coefs-coefs_h_i)

plt.ylabel("absorbance")
plt.xlabel("wavenumber (cm-1)")
plt.legend(("GAAS","HAPI","Error"))
plt.show()
