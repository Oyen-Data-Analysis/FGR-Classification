# You need FILLED MASKS to run this script

import pydicom
import SimpleITK as sitk
import re
import os
from config import SEGMENTATION_PATH, DATA_DIR, OUTPUT_DIR
import cv2
import numpy as np

# Regular expression to match the patient IDs
PATIENT_ID_REGEX = re.compile(r"\d{3}-1")

# IDs of patients whose scans need processing
ids = ['173-1', '175-1', '177-1', '182-1', '183-1', '185-1', '186-1', '187-1', '189-1', '190-1', '192-1', '194-1']

# Function to retrieve the file paths of the scans and masks of the patients listed above
def find_scans_and_masks(directory):
    scans = []
    masks = []
    for root, dirs, files in os.walk(directory):
        id_match = PATIENT_ID_REGEX.search(root)
        if id_match and id_match.group() in ids:
            for f in files:
                if f.endswith('.dcm'):
                    projected_segmentation_path = os.path.join(SEGMENTATION_PATH, f.replace('.dcm', '_filled.mha'))
                    if os.path.exists(projected_segmentation_path):
                        scans.append(os.path.join(root, f))
                        masks.append(projected_segmentation_path)
                        print(f"Found scan and mask for patient {id_match.group()} scan {f}")
    return scans, masks

# Function to apply a segmentation mask to a DICOM image and compute features
def apply_mask_and_segment(scan_path, label_path):
    # Read and format the scan image
    scan_image_array = pydicom.dcmread(scan_path).pixel_array
    scan_image_2d = cv2.cvtColor(scan_image_array, cv2.COLOR_BGR2GRAY)
    # Read and format the mask image
    mask_image_array = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    mask_image_2d = mask_image_array[0] if mask_image_array.ndim > 2 else mask_image_array
    mask_image_2d = mask_image_2d.astype(np.uint8)
    _, binary_mask = cv2.threshold(mask_image_2d, 0, 255, cv2.THRESH_BINARY)
    # Reshape mask if necessary
    if binary_mask.shape != scan_image_2d.shape:
        binary_mask = cv2.resize(binary_mask, (scan_image_2d.shape[1], scan_image_2d.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Apply mask to scan image
    segmented_image = cv2.bitwise_and(scan_image_2d, scan_image_2d, mask=binary_mask)
    return segmented_image

def save_segmented_image(segmented_image, scan_path, output_dir):
    image_filename = os.path.basename(scan_path).replace('.dcm', '_segmented.jpg')
    cv2.imwrite(os.path.join(output_dir, image_filename), segmented_image)

# Run it
scans, masks = find_scans_and_masks(DATA_DIR)
for scan, mask in zip(scans, masks):
    segmented_image = apply_mask_and_segment(scan, mask)
    save_segmented_image(segmented_image, scan, OUTPUT_DIR)