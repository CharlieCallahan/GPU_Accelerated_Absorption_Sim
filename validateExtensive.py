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

#from ssl import ALERT_DESCRIPTION_DECODE_ERROR

import gaas as gs
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import hapi

cwd = os.path.dirname(os.path.realpath(__file__))
if sys.platform == 'win32' or sys.platform == 'win64':
    gaasDirPath = cwd + "\\gaasDir\\"
else:
    gaasDirPath = cwd + "/gaasDir/"
    
if (not os.path.isdir(gaasDirPath)):
    #need to make gaas directory
    os.mkdir(gaasDirPath)

def getError(nus_h,coefs_h,nus_g,coefs_g): #gets percent error
    err = 0
    sum = 0
    length = min(len(coefs_g),len(coefs_h))
    for i in range(length):
        err+=abs(coefs_h[i]-coefs_g[i])
        sum+=coefs_h[i]
    return err/sum*100 #%

def genError(T,P,conc,wavenumStep,startWavenum, endWavenum, mol, iso, gaasDir, id="test"):
    #generates error between HAPI and GAAS and returns execution time

    #for some reason the first run is always slower, so this just primes the GPU so that the timing is accurate for timed runs.
    nus_g32,coefs_g32 = gs.gaasSimVoigt(T, P,conc,wavenumStep,startWavenum,endWavenum,gaasDir,mol,iso,id)

    print("Running GAAS 32")
    t1 = time.time()
    nus_g32,coefs_g32 = gs.gaasSimVoigt(T, P,conc,wavenumStep,startWavenum,endWavenum,gaasDir,mol,iso,id)
    gTime32 = time.time() - t1

    print("Running GAAS 64")
    t1 = time.time()
    nus_g64,coefs_g64 = gs.gaasSimVoigt(T, P,conc,wavenumStep,startWavenum,endWavenum,gaasDir,mol,iso,id)
    gTime64 = time.time() - t1

    print("Running HAPI")
    t1 = time.time()
    nus_h,coefs_h = gs.runHAPI(T, P, conc, wavenumStep, startWavenum, endWavenum, mol, iso,cwd+'\\HTData')
    hTime = time.time() - t1

    return getError(nus_h,coefs_h,nus_g32,coefs_g32), getError(nus_h,coefs_h,nus_g64,coefs_g64), hTime, gTime32, gTime64

def valSpecies(T, P, conc, res, iso, startWavenum, endWavenum, speciesList, gaasDirPath):
    out = []
    for species in speciesList:
        #gs.gaasInit(startWavenum,endWavenum,species,iso,gaasDirPath,cwd+'HTData',"test",loadFromHITRAN=True)
        err32, err64, hTime, gTime32, gTime64 = genError(T,P,conc,res,startWavenum, endWavenum, species, iso, gaasDirPath)
        row = [P,T,conc,species,hTime,gTime32,gTime64,err32,err64]
        print(row)
        out.append(row)
    outpd = pd.DataFrame(out, columns=["P","T","conc","molID","HAPI_Time_s","GAAS32_Time_s","GAAS64_Time_s","g32 Error %","g64 Error %"])
    return outpd

def valConcs(T, P, concs, res, iso, startWavenum, endWavenum, species, gaasDirPath):
    out = []
    for c in concs:
        #gs.gaasInit(startWavenum,endWavenum,species,iso,gaasDirPath,cwd+'HTData',"test",loadFromHITRAN=True)
        err32, err64, hTime, gTime32, gTime64 = genError(T,P,c,res,startWavenum, endWavenum, species, iso, gaasDirPath)
        row = [P,T,c,species,hTime,gTime32,gTime64,err32,err64]
        print(row)
        out.append(row)
    outpd = pd.DataFrame(out, columns=["P","T","conc","molID","HAPI_Time_s","GAAS32_Time_s","GAAS64_Time_s","g32 Error %","g64 Error %"])
    return outpd

def genSpecVal():
    #generates validation dataset over a few species and 4 different conditions: 300K 1atm, 600K 1atm, 300K 5atm, 600K 5atm
    #may take a while
    species =  ['H2O', 'CO2','O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3']
    startWavenum = 1000
    endWavenum = 5000
    wavenumStep = 0.0001 #wavenums per simulation step
    for s in species:
        gs.gaasInit(startWavenum,endWavenum,s,1,gaasDirPath,cwd+'\\HTData',"test",loadFromHITRAN=True)

    iso = 1 #isotopologue num
    T = 300 #K
    P = 1.0 #atm
    conc = 0.05
    results = valSpecies(T,P,conc,wavenumStep, iso,startWavenum,endWavenum,species,gaasDirPath)
    print(results)
    results.to_csv(cwd+"\\Validation\\speciesValidation_300K_1atm.csv")

    iso = 1 #isotopologue num
    T = 600 #K
    P = 1.0 #atm
    conc = 0.05
    results = valSpecies(T,P,conc,wavenumStep, iso,startWavenum,endWavenum,species,gaasDirPath)
    print(results)
    results.to_csv(cwd+"\\Validation\\speciesValidation_600K_1atm.csv")

    iso = 1 #isotopologue num
    T = 300 #K
    P = 5.0 #atm
    conc = 0.05
    results = valSpecies(T,P,conc,wavenumStep, iso,startWavenum,endWavenum,species,gaasDirPath)
    print(results)
    results.to_csv(cwd+"\\Validation\\speciesValidation_300K_5atm.csv")

    iso = 1 #isotopologue num
    T = 600 #K
    P = 5.0 #atm
    conc = 0.05
    results = valSpecies(T,P,conc,wavenumStep, iso,startWavenum,endWavenum,species,gaasDirPath)
    print(results)
    results.to_csv(cwd+"\\Validation\\speciesValidation_600K_5atm.csv")

def genConcVal():
    T = 300
    P = 1.0
    species =  'H2O'
    startWavenum = 1000
    endWavenum = 5000
    wavenumStep = 0.0001 #wavenums per simulation step
    gs.gaasInit(startWavenum,endWavenum,species,1,gaasDirPath,cwd+'\\HTData',"test",loadFromHITRAN=True)

    concs = np.linspace(0.01,0.9,10)
    iso = 1 #isotopologue num
    T = 300 #K
    P = 1.0 #atm
    res = valConcs(T, P, concs, wavenumStep, iso, startWavenum, endWavenum, species, gaasDirPath)
    print(res)
    res.to_csv(os.path.join(cwd,"Validation","concValidation_300K_1atm.csv"))

    iso = 1 #isotopologue num
    T = 600 #K
    P = 1.0 #atm
    res = valConcs(T, P, concs, wavenumStep, iso, startWavenum, endWavenum, species, gaasDirPath)
    print(res)
    res.to_csv(os.path.join(cwd,"Validation","concValidation_600K_1atm.csv"))

    iso = 1 #isotopologue num
    T = 300 #K
    P = 5.0 #atm
    res = valConcs(T, P, concs, wavenumStep, iso, startWavenum, endWavenum, species, gaasDirPath)
    print(res)
    res.to_csv(os.path.join(cwd,"Validation","concValidation_300K_5atm.csv"))

    iso = 1 #isotopologue num
    T = 600 #K
    P = 5.0 #atm
    res = valConcs(T, P, concs, wavenumStep, iso, startWavenum, endWavenum, species, gaasDirPath)
    print(res)
    res.to_csv(os.path.join(cwd,"Validation","concValidation_600K_5atm.csv"))
    
    

def getNumLines(moleculeID, minwvn, maxwvn):
    #returns the number of lines over minwvn to maxwvn
    nu = hapi.getColumns(moleculeID,['nu'])[0]
    numFeats = 0
    for i in range(len(nu)):
        if (nu[i] >= minwvn and nu[i] <= maxwvn):
            numFeats+=1
    return numFeats

def valTempPressure(Temps, Pressures, conc, res, iso, startWavenum, endWavenum, species, gaasDirPath):
    out = []
    for T in Temps:
        for P in Pressures:
            err32, err64, hTime, gTime32, gTime64 = genError(T,P,conc,res,startWavenum, endWavenum, species, iso, gaasDirPath)
            row = [P,T,conc,species,hTime,gTime32,gTime64,err32,err64]
            print(row)
            out.append(row)
    outpd = pd.DataFrame(out, columns=["P","T","conc","molID","HAPI_Time_s","GAAS32_Time_s","GAAS64_Time_s","g32 Error %","g64 Error %"])

    return outpd

def genTiming(T, Pressures, conc, wavenumStep, iso, startWavenum, endWavenums, species, gaasDirPath):
    out = []
    for p in Pressures:
        for endWavenum in endWavenums:
            gs.gaasInit(startWavenum,endWavenum,species,1,gaasDirPath,cwd+'\\HTDataWater',"test",loadFromHITRAN=False)
            nlines = getNumLines(species,startWavenum-gs.WAVENUMBUFFER,endWavenum+gs.WAVENUMBUFFER)
            print("# lines", nlines)
            err32, err64, hTime, gTime32, gTime64 = genError(T,p,conc,wavenumStep,startWavenum, endWavenum, species, iso, gaasDirPath)
            row = [p,T,conc,species,hTime,gTime32,gTime64,err32,err64,nlines]
            print(row)
            out.append(row)
            outpd = pd.DataFrame(out, columns=["P","T","conc","molID","HAPI_Time_s","GAAS32_Time_s","GAAS64_Time_s","g32 Error %","g64 Error %","Num Lines"])

    return outpd

def genTPVal():
    #runs validation over temperature and pressure
    species =  'H2O'
    startWavenum = 1000
    endWavenum = 3000
    wavenumStep = 0.001 #wavenums per simulation step
    iso = 1 #isotopologue num
    conc = 0.05
    gs.gaasInit(startWavenum,endWavenum,species,1,gaasDirPath,cwd+'\\HTData',"test",loadFromHITRAN=True)
    temps = np.linspace(200,1500,10)
    pressures = np.linspace(.01,5,30)
    out = valTempPressure(temps,pressures,conc,wavenumStep,iso,startWavenum,endWavenum,species,gaasDirPath)
    out.to_csv(cwd+"\\Validation\\TPValidationH2O.csv")

def runSpeedTests():
    species =  'H2O'
    startWavenum = 2000
    endWavenum = 4000
    wavenumStep = 0.001 #wavenums per simulation step
    iso = 1 #isotopologue num
    conc = 0.05
    hapi.db_begin(cwd+"/HTDataWater")
    gs.gaasInit(startWavenum,endWavenum,species,1,gaasDirPath,cwd+'\\HTDataWater',"test",loadFromHITRAN=True)
    endWavenums1 = np.linspace(startWavenum+20,startWavenum+(endWavenum-startWavenum)/10,10)
    endWavenums2 = np.linspace(startWavenum+20,endWavenum,10)

    endWavenums = np.concatenate((endWavenums1,endWavenums2))
    pressures = np.linspace(0.1,50,5)
    out = genTiming(300, pressures, conc, wavenumStep, iso, startWavenum, endWavenums, species, gaasDirPath)
    out.to_csv(cwd+"\\Validation\\timing_hp.csv")

def mergeHITEMPFiles(orderedFileList, outfilename):
    #combines hitran .par files into a single monolithic par file
    #used for broadband hitemp simulations
    print("merging HITEMP files.")
    outFile = open(outfilename,'w')
    for file in orderedFileList:
        print(file)
        f = open(file,'r')
        while True:
            line=f.readline()
            if not line:
                break
            outFile.write(line)
        f.close()
    outFile.close()

#gs.gaasInit(1300,3500,'H2O',1,"C:/Users/Charlie/Desktop/HITEMP_DELETE_THIS/db/","C:/Users/Charlie/Desktop/HITEMP_DELETE_THIS/db/","test",loadFromHITRAN=False)
# htd = "C:/Users/Charlie/Desktop/HITEMP_DELETE_THIS/"
# files = [htd+"01_1300-1500_HITEMP2010.par",
#         htd+"01_1500-1750_HITEMP2010.par",
#         htd+"01_1750-2000_HITEMP2010.par",
#         htd+"01_2000-2250_HITEMP2010.par",
#         htd+"01_2250-2500_HITEMP2010.par",
#         htd+"01_2500-2750_HITEMP2010.par",
#         htd+"01_2750-3000_HITEMP2010.par",
#         htd+"01_3000-3250_HITEMP2010.par",
#         htd+"01_3250-3500_HITEMP2010.par"]     
# mergeHITEMPFiles(files, htd+"merged.par")        
#initialize
#hapi.db_begin(cwd+"/HTData")
#for s in species:
#    gs.gaasInit(startWavenum,endWavenum,s,1,gaasDirPath,cwd+'\\HTData',"test",loadFromHITRAN=False)
#genSpecVal()
# genTPVal()
genConcVal()
# runSpeedTests()
#compares output between GAAS and HAPI over a range of different conditions
