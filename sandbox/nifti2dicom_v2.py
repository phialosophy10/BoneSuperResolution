import os
import pydicom
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset
from PIL import Image
import numpy as np

def tiff_volume_to_dicom(tiff_folder, dicom_folder):
    # Create a new DICOM series
    series_instance_uid = generate_uid()
    series_description = "TIFF Volume to DICOM Conversion"
    
    # Create output folder if it doesn't exist
    if not os.path.exists(dicom_folder):
        os.makedirs(dicom_folder)
    
    # List all TIFF files in the folder
    tiff_files = sorted([f for f in os.listdir(tiff_folder) if f.endswith('.tif')])

    for i, tiff_file in enumerate(tiff_files):
        # Read the TIFF image
        tiff_image = Image.open(os.path.join(tiff_folder, tiff_file))
        tiff_array = np.array(tiff_image)
        
        # Create a DICOM dataset
        dicom_dataset = Dataset()
        dicom_dataset.PatientName = "Anonymous"
        dicom_dataset.Modality = "micro CT"  # Other
        dicom_dataset.SeriesInstanceUID = series_instance_uid
        dicom_dataset.SeriesDescription = f"{series_description} - Slice {i + 1}"
        dicom_dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
        
        # Set image attributes
        dicom_dataset.Rows, dicom_dataset.Columns = tiff_array.shape
        dicom_dataset.BitsAllocated = 8
        dicom_dataset.BitsStored = 8
        dicom_dataset.HighBit = 7
        dicom_dataset.PixelRepresentation = 0  # Unsigned integer

        # Convert the image array to bytes
        pixel_data = tiff_array.tobytes()

        dicom_dataset.PixelData = pixel_data

        # Save the DICOM file
        dicom_file = os.path.join(dicom_folder, f"Slice_{i + 1}.dcm")
        dicom_dataset.save_as(dicom_file)

        print(f"Slice {i + 1} converted and saved as DICOM: '{dicom_file}'")

if __name__ == "__main__":
    tiff_folder = "input_tiff_volume"  # Replace with the folder containing TIFF slices
    dicom_folder = "output_dicom"  # Replace with the folder where you want to save the DICOM files
    
    tiff_volume_to_dicom(tiff_folder, dicom_folder)