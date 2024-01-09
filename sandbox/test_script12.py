# %%
import os, math, sys
import numpy as np
import glob

# %%
femur_no = "001"
dataset_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/"
hr_paths_train = sorted(glob.glob(dataset_path + "femur_" + femur_no + "/micro/patches/f_" + femur_no + "_????*.*")) # \d\d\d t

# %%
