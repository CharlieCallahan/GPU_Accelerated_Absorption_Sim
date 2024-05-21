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

import os

os.environ["GAAS_OCL_DEVICE"] = "0"  # set to the device number you want to use if you
# have multiple GPUs
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import published_validation.hapiValidationFunctions as hvf
import gaas_ocl as gs

startWavenum = 5980
endWavenum = 6150
wavenumStep = 0.003  # wavenums per simulation step
mol = "CH4_HITEMP"  # molecule name
iso = 1  # isotopologue num
T = 3000  # K
P = 1.0  # atm
conc = 0.02
# pathlength is assumed to be 1cm, if you want to use a different pathlenth, scale the absorbance by pl_cm

cwd = os.path.dirname(os.path.realpath(__file__))
if sys.platform == "win32" or sys.platform == "win64":
    gaasDirPath = cwd + "\\gaasDir\\"
else:
    gaasDirPath = cwd + "/gaasDir/"

if sys.platform == "win32" or sys.platform == "win64":
    dbdir = cwd + "\\DBDir\\"
else:
    dbdir = cwd + "/DBDir/"

###########
dbdir = r"C:\git\linelists\linelists_CH4"
###########

if not os.path.isdir(gaasDirPath):
    # need to make gaas directory
    os.mkdir(gaasDirPath)

if not os.path.isdir(dbdir):
    # need to make database directory
    os.mkdir(dbdir)

# initialize GAAS and create GAAS database file for the wavenumber range and molecule you want to simulate
# This needs to be called once for each molecule/isotope you want to simulate.
# Once GAAS binary files are generated, they can be re-used by different runs for the same molecule/isotope.
absDB = gs.gen_abs_db(
    mol, iso, startWavenum, endWavenum, dbdir, 0, loadFromHITRAN=False
)
# (tempK, pressureAtm, conc,  wavenumRes, startWavenum, endWavenum, moleculeID)

print("Running HAPI simulation...\n")
t0 = time.time()
nus_h, coefs_h = hvf.runHAPI(
    T, P, conc, wavenumStep, startWavenum, endWavenum, mol, iso, "DBDir"
)
t0_h = time.time()

if mol not in gs.getHITRANMolecules():
    molecule_list = []
    [
        molecule_list.append(molec)
        for idx, molec in enumerate(gs.getHITRANMolecules())
        if molec in mol
    ]
    mol = molecule_list[0]

t1 = time.time()
nus, coefs = gs.simVoigt(
    T,
    P,
    conc,
    wavenumStep,
    startWavenum,
    endWavenum,
    mol,
    iso,
    absDB,
    gs.get_tips_calc(mol, iso),
)

print("GAAS sim time: ", time.time() - t1)
print("HAPI sim time: ", t0_h - t0)


def getError(nus_h, coefs_h, nus_g, coefs_g):
    err = 0
    sum = 0
    for i in range(len(coefs_h)):
        err += abs(coefs_h[i] - coefs_g[i])
        sum += coefs_h[i]
    return err / sum


print("Simulation error = ", getError(nus_h, coefs_h, nus, coefs) * 100, "%")

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(nus, coefs)
ax1.plot(nus_h, coefs_h)
ax1.legend(("GAAS", "HAPI"))
ax1.set_ylabel("Absorbance")

ax2.plot(nus, coefs_h - coefs)
ax2.set_ylabel("Absolute Error")

plt.xlabel("wavenumber (cm-1)")
plt.show()
