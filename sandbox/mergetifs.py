# %%
import os
import tifftools

# %%
tiff_files_li=[]
root_path = '/work3/soeba/HALOS/Data/microCT'
tif_path = root_path + 'femur_74_merged_tiff/' #01 15 21 74
for ti in os.listdir(tif_path):
    if '.tif' in ti:
        tiff_files_li.append(tif_path+ti)

# %%
tifftools.tiff_concat(tiff_files_li, root_path+"femur74.tiff", overwrite=True) #01 15 21 74