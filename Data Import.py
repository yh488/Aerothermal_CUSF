# from google.colab import drive
# import os
# drive.mount('/content/drive')
#
# #Comment out the first 2 lines if you've already selected the right directory
# os.chdir("drive/Shareddrives/Cambridge University Spaceflight")
# os.chdir("108 Griffin I/01 Docs/03 Aerodynamics/Nosecone development/Aerothermal modelling")
# os.listdir()

import pandas as pd
from pandas import ExcelFile
Path = '/Users/olivahuang/CUSF/Griffin_Prelim_Flight_Profile.xlsx'
csvfile = "Griffin_Prelim_Flight_Profile.xlsx"

pd_csv = pd.read_csv(csvfile)
time_list = np.array(pd_csv.loc[:, "# Time (s)"][::5])
altitude_list = np.array(pd_csv.loc[:, " Z (m)"][::5])
speed_list = np.array([
    np.linalg.norm([pd_csv.loc[:, " Vx (m/s)"][i], pd_csv.loc[:, " Vy (m/s)"][i], pd_csv.loc[:, " Vz (m/s)"][i]])
    for i in range(len(pd_csv.loc[:, " Vx (m/s)"]))
    ][::5])

trajectory_data = {
    "time": time_list,
    "altitude": altitude_list,
    "speed": speed_list
}

plt.plot(trajectory_data["time"], trajectory_data["altitude"])
plt.plot(trajectory_data["time"], trajectory_data["speed"])