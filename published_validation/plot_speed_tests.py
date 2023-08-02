import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from math import nextafter, inf
cwd = os.path.dirname(os.path.realpath(__file__))
plt.style.use(cwd+'\\eli_default.mplstyle')

htp_speed_val = pd.read_csv(cwd+"\\voigt_speed_val_new_adaptive.csv")
plt.plot(htp_speed_val["numFeats"], np.array(htp_speed_val["HAPITime"])/np.array(htp_speed_val["gaasTime"])*100 )
plt.xlabel("# of lines")
plt.ylabel("Speed increase (%)")
plt.title("GAAS HTP Performance")
plt.show()
htp_df = pd.read_csv(cwd+"\\htp_speed_gaas.csv")
plt.scatter(htp_df["numFeats"], htp_df["numFeats"]/np.array(htp_df["gaasTime"]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
# plt.gca()._get_lines.color_cycle

plt.plot(htp_speed_val["numFeats"], htp_speed_val["numFeats"]/htp_speed_val["HAPITime"])

plt.yscale("log")
plt.xlabel("# of lines")
plt.ylabel("Lines per second")
plt.legend(("GAAS", "HAPI"))
plt.title("GAAS HTP Performance")
plt.show()

voigt_speed_val = pd.read_csv(cwd+"\\voigt_speed_val.csv")
plt.plot(voigt_speed_val["numFeats"], np.array(voigt_speed_val["HAPITime"])/np.array(voigt_speed_val["gaasTime"])*100 )
plt.xlabel("# of lines")
plt.ylabel("Speed increase (%)")
plt.title("GAAS Voigt Performance")
plt.show()
voigt_df = pd.read_csv(cwd+"\\voigt_speed_gaas.csv")
plt.scatter(voigt_df["numFeats"], voigt_df["numFeats"]/np.array(voigt_df["gaasTime"]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
# plt.gca()._get_lines.color_cycle

plt.plot(voigt_speed_val["numFeats"], voigt_speed_val["numFeats"]/voigt_speed_val["HAPITime"])

plt.yscale("log")
plt.xlabel("# of lines")
plt.ylabel("Lines per second")
plt.legend(("GAAS", "HAPI"))
plt.title("GAAS Voigt Performance")
plt.show()