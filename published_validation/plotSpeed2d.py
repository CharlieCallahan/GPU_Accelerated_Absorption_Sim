from turtle import speed
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys
import numpy as np

if (len(sys.argv) < 2):
    print("Error, no arguments. Input timing output file as arg to plot results.")
    print("Args: filename, (gaas time filename)")
    exit(-1)

fname = sys.argv[1]
data = pd.read_csv(fname)
p = data['meanHWHM']*2000
tGaas = data['gaasTime']
tHapi = data['HAPITime']
nlines = data['numFeats']

if(len(sys.argv) > 2):
    gaas_data = pd.read_csv(sys.argv[2])
    tGaas = gaas_data['gaasTime']
    print(tGaas)
speedup = np.array(tHapi)/np.array(tGaas)
res_nl = 10
res_hwhm = 10

fig = plt.figure()
axes = fig.gca(projection ='3d')

x,y = np.meshgrid(nlines[:res_nl],  p[range(0,res_hwhm*res_nl,res_nl)])
print(x)
print(y)
speedup_a = np.ndarray((res_nl,res_hwhm))
for i in range(res_hwhm): #p
    for j in range(res_nl): #nlines
        speedup_a[j,i] = speedup[i*res_nl+j]
print(x.shape)
print(y.shape)

axes.plot_surface(x.transpose(), y.transpose() , speedup_a,cmap=cm.coolwarm, linewidth=1.0, antialiased=False)
axes.plot_wireframe(x.transpose(), y.transpose() , speedup_a, linewidth=1.0, rstride=1, cstride=1, color='k')

plt.ylabel("FWHM / (Spectral Resolution)")
plt.xlabel("Number of lines")
plt.title("Hartmann Tran")
axes.set_zlabel("Speedup", labelpad = 10)
plt.show()

# Pdim = 0
# for t in T:
#     if t==T[0]:
#         Pdim+=1

# Tdim =int( len(T)/Pdim)

# err32 = data['g32 Error %']
# err64 = data['g64 Error %']
# outArr32 = np.ndarray((Tdim,Pdim))
# outArr64 = np.ndarray((Tdim,Pdim))

# for i in range(Tdim):
#     for j in range(Pdim):
#         outArr32[i,j] = err32[i*Pdim+j]
#         outArr64[i,j] = err64[i*Pdim+j]
# pAdj = p[:Pdim]
# TAdj = []
# for i in range(0,len(T),Tdim):
#     TAdj.append(T[i])
# x,y = np.meshgrid(pAdj, TAdj)
# fig = plt.figure()
# axes = fig.gca(projection ='3d')
# axes.plot_surface(x, y, outArr32)
# plt.ylabel("Temperature (K)")
# plt.xlabel("Pressure (atm)")
# axes.set_zlabel("Error (%)", labelpad = 10)
# plt.show()

