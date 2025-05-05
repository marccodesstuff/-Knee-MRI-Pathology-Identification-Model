import datetime
import os
from pathlib import Path
import argparse
import h5py
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
import xmltodict

def fastmri_to_dicom(filename: Path,
    reconstruction_name: str,
    outfolder: Path,
    flip_up_down: bool = False,
    flip_left_right: bool = False) -> None:

    fileparts = os.path.splitext(filename.name)
    patientName = fileparts[0]
    try:
        f = h5py.File(filename,'r')
    except Exception as e:
        print(f"Error opening file {filename}: {e}")
        return

    # Create a subfolder for each H5 file's DICOMs within the main output folder
    patient_outfolder = outfolder / patientName
    patient_outfolder.mkdir(parents=True, exist_ok=True)

    if 'ismrmrd_header' not in f.keys():
        print(f'ISMRMRD header not found in file {filename}')
        f.close()
        return

    if reconstruction_name not in f.keys():
        print(f'Reconstruction name {reconstruction_name} not found in file {filename}')
        f.close()
        return

    # Get some header information
    try:
        head = xmltodict.parse(f['ismrmrd_header'][()])
        reconSpace = head['ismrmrdHeader']['encoding']['reconSpace'] # ['matrixSize', 'fieldOfView_mm']
        measurementInformation = head['ismrmrdHeader']['measurementInformation'] # ['measurementID', 'patientPosition', 'protocolName', 'frameOfReferenceUID']
        acquisitionSystemInformation = head['ismrmrdHeader']['acquisitionSystemInformation'] # ['systemVendor', 'systemModel', 'systemFieldStrength_T', 'relativeReceiverNoiseBandwidth' 'receiverChannels', 'coilLabel', 'institutionName']
        H1resonanceFrequency_Hz = head['ismrmrdHeader']['experimentalConditions']['H1resonanceFrequency_Hz']
        sequenceParameters = head['ismrmrdHeader']['sequenceParameters'] # ['TR', 'TE', 'TI', 'flipAngle_deg', 'sequence_type', 'echo_spacing']
    except Exception as e:
        print(f"Error parsing header in file {filename}: {e}")
        f.close()
        return

    # Some calculated values
    try:
        pixelSizeX = float(reconSpace['fieldOfView_mm']['x'])/float(reconSpace['matrixSize']['x'])
        pixelSizeY = float(reconSpace['fieldOfView_mm']['y'])/float(reconSpace['matrixSize']['y'])
        sliceThickness = float(reconSpace['fieldOfView_mm']['z'])
    except KeyError as e:
        print(f"Missing expected key in reconSpace for file {filename}: {e}")
        f.close()
        return
    except ValueError as e:
        print(f"Error converting header values to float for file {filename}: {e}")
        f.close()
        return


    # Get and prep pixel data
    img_data = f[reconstruction_name][:]
    f.close() # Close the H5 file as soon as data is read
    slices = img_data.shape[0]

    if flip_left_right:
        img_data = img_data[:, :, ::-1]

    if flip_up_down:
        img_data = img_data[:, ::-1, :]

    image_max = 1024
    # Use np.nanpercentile to handle potential NaN values if data quality varies
    p999 = np.nanpercentile(img_data, 99.9)
    if p999 == 0: # Avoid division by zero if percentile is 0
        scale = 1.0
    else:
        scale = image_max / p999

    pixels_scaled = np.clip((scale * img_data), 0, image_max).astype('int16')
    p01 = np.nanpercentile(pixels_scaled, 0.1)
    p999_scaled = np.nanpercentile(pixels_scaled, 99.9)
    windowWidth = 2 * (p999_scaled - p01)
    if windowWidth <= 0: # Ensure window width is positive
        windowWidth = image_max
    windowCenter = p01 + windowWidth/2

    studyInstanceUid = generate_uid('999.')
    seriesInstanceUid = generate_uid('9999.')

    print(f"Processing {filename.name}: {slices} slices...")
    for s in range(0, slices):
        slice_filename = f"{patientName}_{s:03d}.dcm"
        slice_full_path = patient_outfolder / slice_filename
        slice_pixels = pixels_scaled[s,:,:]

        # File meta info data elements
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4' # MR Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid('9999.') # Unique per file
        file_meta.ImplementationClassUID = generate_uid('9999.99') # Your implementation UID
        file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1' # Explicit VR Little Endian

        # Main data elements
        ds = Dataset()

        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        timeStr = dt.strftime('%H%M%S.%f')[:-3] # DICOM time format
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID # Should match meta header
        ds.ContentTime = timeStr
        ds.Modality = 'MR'
        # ds.ModalitiesInStudy = ['', 'PR', 'MR', ''] # Optional, often derived
        ds.StudyDescription = measurementInformation.get('protocolName', 'Unknown Protocol')
        ds.PatientName = patientName
        ds.PatientID = patientName
        ds.PatientBirthDate = '' # Avoid setting fake data if unknown
        ds.PatientSex = '' # Avoid setting fake data if unknown
        ds.PatientAge = '' # Avoid setting fake data if unknown
        ds.PatientIdentityRemoved = 'YES'
        ds.MRAcquisitionType = '2D' # Assuming 2D based on typical fastMRI data
        ds.SequenceName = sequenceParameters.get('sequence_type', 'Unknown Sequence')
        ds.SliceThickness = str(sliceThickness)
        ds.RepetitionTime = str(sequenceParameters.get('TR', ''))
        ds.EchoTime = str(sequenceParameters.get('TE', ''))
        ds.ImagingFrequency = str(H1resonanceFrequency_Hz)
        ds.ImagedNucleus = '1H'
        ds.EchoNumbers = "1" # Assuming single echo
        ds.MagneticFieldStrength = str(acquisitionSystemInformation.get('systemFieldStrength_T', ''))
        # SpacingBetweenSlices might differ from SliceThickness, requires more info if available
        ds.SpacingBetweenSlices = str(sliceThickness)
        ds.FlipAngle = str(sequenceParameters.get('flipAngle_deg', ''))
        ds.PatientPosition = measurementInformation.get('patientPosition', '')
        ds.StudyInstanceUID = studyInstanceUid # Consistent for all slices from this H5
        ds.SeriesInstanceUID = seriesInstanceUid # Consistent for all slices from this H5
        ds.StudyID = measurementInformation.get('measurementID', patientName) # Use patientName as fallback
        ds.InstanceNumber = str(s+1)
        ds.ImagesInAcquisition = str(slices) # Total slices in this series
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.NumberOfFrames = "1"
        ds.Rows = slice_pixels.shape[0]
        ds.Columns = slice_pixels.shape[1]
        ds.PixelSpacing = [str(pixelSizeX), str(pixelSizeY)]
        # ds.PixelAspectRatio = [1, 1] # Often derived from PixelSpacing
        ds.BitsAllocated = 16
        ds.BitsStored = 16 # Store full precision
        ds.HighBit = 15
        ds.PixelRepresentation = 1 # Signed integer
        ds.SmallestImagePixelValue = int(np.min(slice_pixels))
        ds.LargestImagePixelValue = int(np.max(slice_pixels))
        ds.BurnedInAnnotation = 'NO'
        ds.WindowCenter = str(windowCenter)
        ds.WindowWidth = str(windowWidth)
        ds.LossyImageCompression = '00' # No lossy compression
        # ds.StudyStatusID = 'COMPLETED' # Optional
        # ds.ResultsID = '' # Optional

        ds.PixelData = slice_pixels.tobytes()

        ds.file_meta = file_meta
        ds.is_implicit_VR = False # Use Explicit VR
        ds.is_little_endian = True
        try:
            ds.save_as(slice_full_path, write_like_original=False)
        except Exception as e:
            print(f"Error saving DICOM file {slice_full_path}: {e}")
    print(f"Finished processing {filename.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert fastMRI H5 files in a directory to DICOMs')
    parser.add_argument('--infolder' , type=str, help='Input folder containing H5 files', required=True)
    parser.add_argument('--reconstruction_name' , type=str, help='Reconstruction name within H5 files', default='reconstruction_rss', required=False)
    parser.add_argument('--outfolder', type=str, help='Main output folder for DICOM subdirectories', required = True)
    # Changed boolean args to store_true/store_false for standard CLI behavior
    parser.add_argument('--flip_up_down', action='store_true', help='Flip image up/down (default: False)')
    parser.add_argument('--no_flip_up_down', dest='flip_up_down', action='store_false', help='Do not flip image up/down')
    parser.set_defaults(flip_up_down=False) # Default is not to flip
    parser.add_argument('--flip_left_right', action='store_true', help='Flip image left/right (default: False)')
    parser.add_argument('--no_flip_left_right', dest='flip_left_right', action='store_false', help='Do not flip image left/right')
    parser.set_defaults(flip_left_right=False) # Default is not to flip

    args = parser.parse_args()

    infolder_path = Path(args.infolder)
    outfolder_path = Path(args.outfolder)

    if not infolder_path.is_dir():
        print(f"Error: Input folder '{args.infolder}' not found or is not a directory.")
        return

    # Ensure the main output directory exists
    outfolder_path.mkdir(parents=True, exist_ok=True)

    # Find all .h5 files in the input directory
    h5_files = list(infolder_path.glob('*.h5'))
    if not h5_files:
        print(f"No .h5 files found in '{args.infolder}'.")
        return

    print(f"Found {len(h5_files)} H5 files to process.")

    # Process each H5 file
    for h5_file in h5_files:
        print(f"\nStarting processing for: {h5_file.name}")
        fastmri_to_dicom(filename = h5_file,
            reconstruction_name=args.reconstruction_name,
            outfolder=outfolder_path, # Pass the main output folder
            flip_up_down=args.flip_up_down,
            flip_left_right=args.flip_left_right)

    print("\nBatch processing complete.")

if __name__ == '__main__':
    main()