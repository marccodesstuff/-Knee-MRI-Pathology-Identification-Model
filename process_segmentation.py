import os
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bar

# Define the path to the output_folder
output_folder = 'output_folder'
annotations = pd.read_csv('annotation.csv')  # Load your annotations CSV file

# Define the path for saving the .npy file
segmentation_data_file = 'segmentation_data.npy'

# Check if the .npy file already exists
if os.path.exists(segmentation_data_file):
    # Load the segmentation data from the .npy file
    segmentation_data = np.load(segmentation_data_file, allow_pickle=True)
    print(f"✅ Loaded segmentation data from {segmentation_data_file}")
else:
    # Prepare dataset
    segmentation_data = []
    files_list = [f for f in os.listdir(output_folder) if f.endswith('.dcm')]

    # Using tqdm to track progress
    for filename in tqdm(files_list, desc="Processing DICOM Segmentation"):
        filepath = os.path.join(output_folder, filename)

        # Read the DICOM file
        dicom_file = pydicom.dcmread(filepath)
        pixel_array = dicom_file.pixel_array
        height, width = pixel_array.shape

        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Extract file ID and slice number
        try:
            file_id = filename.split('_')[0]
            slice_number = int(filename.split('_')[1].split('.')[0])

            # Get corresponding annotations
            file_annotations = annotations[(annotations['file'] == file_id) & (annotations['slice'] == slice_number)]

            # Draw bounding boxes on the mask
            for _, row in file_annotations.iterrows():
                x, y, box_width, box_height = row['x'], row['y'], row['width'], row['height']
                mask[y:y+box_height, x:x+box_width] = 1  # Mark region as foreground
        except Exception as e:
            print(f"Error processing {filename}: {e}")

        # Append to segmentation data
        segmentation_data.append({
            'image': pixel_array,
            'mask': mask
        })

    # Save the segmentation data to a .npy file
    np.save(segmentation_data_file, segmentation_data)
    print(f"✅ Saved segmentation data to {segmentation_data_file}")

# Print summary
print(f"✅ Total segmentation samples: {len(segmentation_data)}")
