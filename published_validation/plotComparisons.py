import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from math import nextafter, inf
cwd = os.path.dirname(os.path.realpath(__file__))
plt.style.use(cwd+'\\eli_default.mplstyle')
v_df = pd.read_csv(cwd+"\\voigt_comparison_plot.csv")
htp_df = pd.read_csv(cwd+"\\htp_comparison_plot.csv")

maxVal = np.mean(v_df["coefs_hapi"])
eps16 = np.spacing(maxVal,dtype=np.float16)
eps32 = np.spacing(maxVal,dtype=np.float32)
eps64 = np.spacing(maxVal,dtype=np.float64)
print(eps32)
print("max", np.max(abs(v_df["coefs_gaas"] - v_df["coefs_hapi"])))
fontSize = 16
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : fontSize}

matplotlib.rc('font', **font)

fig, axs = plt.subplots(2,2,sharex=True)
fig.set_size_inches(6, 3)

axs[0][0].set_title("Simulated Voigt Spectrum")
axs[0][0].plot(v_df["wvn_hapi"],v_df["coefs_hapi"])
axs[0][0].plot(v_df["wvn_gaas"],-1*v_df["coefs_gaas"])
axs[0][0].legend(("HAPI", "GAAS"))
axs[0][0].tick_params(axis='both', which='major', labelsize=fontSize)
axs[0][0].tick_params(axis='both', which='minor', labelsize=fontSize)

axs[0][1].set_yscale("log")
axs[0][1].plot(v_df["wvn_hapi"], abs(v_df["coefs_gaas"] - v_df["coefs_hapi"]))
axs[0][1].plot([v_df["wvn_hapi"][0],v_df["wvn_hapi"][len(v_df["wvn_hapi"])-1]],[eps32,eps32] )
axs[0][1].plot([v_df["wvn_hapi"][0],v_df["wvn_hapi"][len(v_df["wvn_hapi"])-1]],[eps64,eps64] )
axs[0][1].legend(("Error", "32 bit precision","64 bit precision" ))
axs[0][1].set_xlabel("Wavenumber (cm-1)", fontsize=fontSize )
axs[0][1].set_ylabel("Error", fontsize=fontSize)
axs[0][1].tick_params(axis='both', which='major', labelsize=fontSize)
axs[0][1].tick_params(axis='both', which='minor', labelsize=fontSize)
axs[0][1].set_xlim((2300,3300))
axs[0][1].set_ylim((1e-16,1e-2))
# plt.show()

#HTP
# fig, axs = plt.subplots(2,1,sharex=True)
# fig.set_size_inches(11.6, 8.2)
maxVal = np.mean(htp_df["coefs_hapi"])
eps16 = np.spacing(maxVal,dtype=np.float16)
eps32 = np.spacing(maxVal,dtype=np.float32)
eps64 = np.spacing(maxVal,dtype=np.float64)
print(eps32)
print("max", np.max(abs(htp_df["coefs_gaas"][:-10] - htp_df["coefs_hapi"][:-10])))


axs[1][0].set_title("Simulated HTP Spectrum")
axs[1][0].plot(htp_df["wvn_hapi"],htp_df["coefs_hapi"])
axs[1][0].plot(htp_df["wvn_gaas"],-1*htp_df["coefs_gaas"])
axs[1][0].legend(("HAPI", "GAAS"))
axs[1][1].set_yscale("log")
axs[1][1].plot(htp_df["wvn_hapi"], abs(htp_df["coefs_gaas"] - htp_df["coefs_hapi"]))
axs[1][1].plot([htp_df["wvn_hapi"][0],htp_df["wvn_hapi"][len(htp_df["wvn_hapi"])-1]],[eps32,eps32] )
axs[1][1].plot([htp_df["wvn_hapi"][0],htp_df["wvn_hapi"][len(htp_df["wvn_hapi"])-1]],[eps64,eps64] )
axs[1][1].legend(("Error", "32 bit precision","64 bit precision" ))
axs[1][1].set_xlabel("Wavenumber (cm-1)")
axs[1][1].set_ylabel("Error")
axs[1][1].set_xlim((2300,3300))
axs[1][1].set_ylim((1e-16,1e-2))

plt.show()