# %% Packages
import nibabel as nib
import pydicom
import os
from skimage import io
#import numpy as np

# %% Set data path
#femur_no_list = ['001']
femur_no_list = ['002','013','015','021','026','031','074','075','083','086','138','164','172']
dicom_path = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/DICOM_micro_CT/dicom.dcm'
save_path = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/DICOM_micro_CT/'

# %% Functions to convert and save as dicom
def convertNsave(arr,file_dir,femur_no='',index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put 
    the name of each slice while using a for loop to convert all the slices
    """
    
    dicom_file = pydicom.dcmread(dicom_path)
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 0
    dicom_file.PixelData = arr.tobytes()
    dicom_file.file_meta.MediaStorageSOPInstanceUID = '1.3.12.2.1' + femur_no + '.5.1.4.9' + str(index).zfill(4) + '.30000022022106245548000045600'
    dicom_file.SOPInstanceUID = '1.3.12.2.1' + femur_no + '.5.1.4.9' + str(index).zfill(4) + '.30000022022106245548000045600'
    dicom_file.save_as(file_dir + femur_no + '_slice_' + str(index) + '.dcm')

def nifti2dicom_1file(nifti_path, out_dir, femur_no):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_path`: the path to the one nifti file
    `out_dir`: the path to output directory
    """

    nifti_file = nib.load(nifti_path)
    nifti_array = nifti_file.get_fdata()
    number_slices = nifti_array.shape[2]
    out_path = os.path.join(out_dir, femur_no)
    out_path = out_dir + femur_no + '/'

    for slice_ in range(number_slices):
        convertNsave(nifti_array[:,:,slice_], out_path, femur_no, slice_)
        
def tiff2dicom_1file(tiff_path, out_dir, femur_no):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_path`: the path to the one nifti file
    `out_dir`: the path to output directory
    """

    vol = io.imread(tiff_path)
    vol = vol.transpose((1, 2, 0))
    number_slices = vol.shape[2]
    out_path = os.path.join(out_dir, femur_no)
    out_path = out_dir + femur_no + '/'

    for slice_ in range(number_slices):
        convertNsave(vol[:,:,slice_], out_path, femur_no, slice_)
        
def nifti2dicom_mfiles(nifti_dir, out_dir=''):
    """
    This function is to convert multiple nifti files into dicom files
    `nifti_dir`: List of file locations of the nifti-files
    `out_dir`: Put the path to where you want to save all the dicoms here.
    PS: Each nifti file's folders will be created automatically, so you do not need to create an empty folder for each patient.
    """

    files = os.listdir(nifti_dir)
    for file in files:
        in_path = os.path.join(nifti_dir, file)
        out_path = os.path.join(out_dir, file)
        os.mkdir(out_path)
        nifti2dicom_1file(in_path, out_path)

# %% Load and save files via functions
# for femur in femur_no_list:
#     nifti_path = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_' + femur + '/micro/volume/f_' + femur + '.nii'
#     nifti2dicom_1file(nifti_path, save_path, femur)
    
for femur in femur_no_list:
    tiff_path = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_' + femur + '/micro/volume/f_' + femur + '.tif'
    tiff2dicom_1file(tiff_path, save_path, femur)