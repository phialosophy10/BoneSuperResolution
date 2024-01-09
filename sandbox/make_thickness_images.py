# %% Packages
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
ps.visualization.set_mpl_style()
import glob

# %% Set data paths
dataset_path = "/work3/soeba/HALOS/Data/Images"

res_type = "hi-res" #"low-res", "hi-res"                  # choose whether to train on low-res or hi-res clinical CT scans
if res_type == "low-res":
    res_type += "/linear" #"/nn", "/linear"

femur_no = "01"

target_path_train = dataset_path + "/train" + res_type + "/thick/m" + femur_no + "_"
target_path_test = dataset_path + "/test" + res_type + "/thick/m" + femur_no + "_"

hr_paths_train = sorted(glob.glob(dataset_path + "/train_"+res_type+"/hr/m"+femur_no+"*.*"))
hr_paths_test = sorted(glob.glob(dataset_path + "/test_"+res_type+"/hr/m"+femur_no+"*.*"))

# %% Load image slices, calculate thickness, save

for i in range(len(hr_paths_train)):
    hr_im = Image.open(hr_paths_train[i])
    hr_im[hr_im<=0.40] = 0
    hr_im[hr_im>0.40] = 1
    thick = ps.filters.local_thickness(hr_im, mode='dt')
    thick.save(target_path_train + str(i).zfill(4) + ".jpg")

for i in range(len(hr_paths_test)):
    hr_im = Image.open(hr_paths_test[i])
    hr_im[hr_im<=0.40] = 0
    hr_im[hr_im>0.40] = 1
    thick = ps.filters.local_thickness(hr_im, mode='dt')
    thick.save(target_path_test + str(i).zfill(4) + ".jpg")