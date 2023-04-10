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

# import gaas as gs
import gaas_ocl as gs
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import htpValidation as htpVal

startWavenum = 7000
endWavenum = 80000
wavenumStep = 0.005 #wavenums per simulation step
molarMass = 1.0
tempK = 300

feat_wavenums = np.linspace(startWavenum,endWavenum,100000)
feat_data = []
for lc in feat_wavenums:
    feat_data.append(gs.HTPFeatureData(lc,0.1,0.1,0.1,0.1,0.1,0.1,1.0))
t0 = time.time()
wvn, spec = gs.simHTP_legacy(feat_data,tempK,molarMass,wavenumStep,startWavenum,endWavenum)
gaasTime = time.time()-t0
# plt.plot(wvn,spec)
# plt.show()
print("gaas time: ",gaasTime)

t0=time.time()
wvnh, spech = htpVal.hapiSimHTP(feat_data,tempK,molarMass,wavenumStep,startWavenum,endWavenum)
hapiTime = time.time()-t0
print("hapi time: ",hapiTime)
print("GAAS is ",hapiTime/gaasTime, " times faster")
plt.plot(spec)
plt.plot(spech)
plt.plot((spec-spech))
plt.legend(("GAAS","HAPI","Residual"))
plt.show()
