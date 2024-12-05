import os
import numpy as np
import pydicom
import SimpleITK as sitk
import cv2
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import csv


# Function to find the largest rectangle close to the target area in a binary mask
def largest_rectangle_with_specific_area(binary_mask, target_area):
    nrows, ncols = binary_mask.shape
    best_rectangle = (None, float('inf'))  # (rectangle, area difference)
    h = np.zeros(ncols, dtype=int)

    # Compute the height of histograms for each column
    for row in range(nrows):
        for col in range(ncols):
            h[col] = h[col] + 1 if binary_mask[row, col] else 0

        # Find the largest rectangle in the row
        for start_col in range(ncols):
            if h[start_col]:
                width = 1
                for k in range(start_col + 1, ncols):
                    if h[k] >= h[start_col]:
                        width += 1
                    else:
                        break
                area = width * h[start_col]
                area_diff = abs(area - target_area)
                if area_diff < best_rectangle[1]:
                    best_rectangle = ((row - h[start_col] + 1, start_col, h[start_col], width), area_diff)

    # Return the coordinates and dimensions of the best rectangle
    top_left_y, top_left_x, height, width = best_rectangle[0]
    return top_left_x, top_left_y, width, height


# Function to find the largest rectangle with a specified area in a segmentation
def find_largest_rectangle_in_segmentation(dicom_path, segmentation_path, target_area):
    dicom_image = sitk.ReadImage(dicom_path)
    dicom_array = sitk.GetArrayFromImage(dicom_image)
    dicom_image_2d = dicom_array[0]  # Assuming it's a single slice for simplicity
    dicom_image_2d = cv2.normalize(dicom_image_2d, None, 0, 255, cv2.NORM_MINMAX)

    segmentation_image = sitk.ReadImage(segmentation_path)
    segmentation_array = sitk.GetArrayFromImage(segmentation_image)
    segmentation_2d = segmentation_array[0]  # Assuming it's a single slice for simplicity

    segmentation_2d = segmentation_2d.astype(np.uint8)
    if len(segmentation_2d.shape) == 3 and segmentation_2d.shape[2] == 3:
        segmentation_2d = cv2.cvtColor(segmentation_2d, cv2.COLOR_BGR2GRAY)
    _, segmentation_2d = cv2.threshold(segmentation_2d, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(segmentation_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dicom_image_2d, contours, -1, (0, 255, 0), 2)  # Draw green contours on the DICOM image

    # Find the largest rectangle close to the target area
    top_left_x, top_left_y, width, height = largest_rectangle_with_specific_area(segmentation_2d, target_area)

    # Draw the rectangle on the DICOM image
    cv2.rectangle(dicom_image_2d, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), (0, 0, 255), 3)
    area = width * height
    return dicom_image_2d, area


# Function to compute GLCM features of an image
def compute_glcm_features(image):
    image = image.astype('uint8')
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0],
    }
    return features


# Function to apply a segmentation mask to a DICOM image and compute features
def apply_mask_and_compute_features(dicom_path, segmentation_path, output_dir, target_area=30000):
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_image_array = dicom_data.pixel_array

    if len(dicom_image_array.shape) == 3 and dicom_image_array.shape[-1] in [3, 4]:
        dicom_image_2d = cv2.cvtColor(dicom_image_array, cv2.COLOR_BGR2GRAY)
    else:
        dicom_image_2d = dicom_image_array

    dicom_image_2d = cv2.normalize(dicom_image_2d, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    segmentation_image = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path))
    segmentation_2d = segmentation_image[0] if segmentation_image.ndim > 2 else segmentation_image
    segmentation_2d = segmentation_2d.astype(np.uint8)
    _, binary_mask = cv2.threshold(segmentation_2d, 0, 255, cv2.THRESH_BINARY)

    # Resize the binary mask if it doesn't match the DICOM image dimensions
    if binary_mask.shape != dicom_image_2d.shape:
        binary_mask = cv2.resize(binary_mask, (dicom_image_2d.shape[1], dicom_image_2d.shape[0]), interpolation=cv2.INTER_NEAREST)

    if len(binary_mask.shape) == 3 and binary_mask.shape[2] == 3:
        binary_mask = binary_mask[:, :, 0]

    # Apply the mask to the DICOM image
    segmented_image = cv2.bitwise_and(dicom_image_2d, dicom_image_2d, mask=binary_mask)
    features = compute_glcm_features(segmented_image)

    average_pixel_intensity = cv2.mean(segmented_image, mask=binary_mask)[0]

    # Save the segmented image
    image_filename = os.path.basename(dicom_path).replace('.dcm', '_segmented.jpg')
    cv2.imwrite(os.path.join(output_dir, image_filename), segmented_image)

    # Find and save the largest rectangle in the segmentation
    outlined_image, area = find_largest_rectangle_in_segmentation(dicom_path, segmentation_path, target_area)
    outlined_filename = os.path.basename(dicom_path).replace('.dcm', '_outlined.jpg')
    cv2.imwrite(os.path.join(output_dir, outlined_filename), outlined_image)

    return features, dicom_path, image_filename, average_pixel_intensity, area, outlined_filename


# Function to process a directory of DICOM and segmentation files and compute GLCM features
def process_directory_for_glcm(dicom_directory, group_name, output_dir="Output_Segmented_Images", target_area=30000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    features_list = []
    for subdir, dirs, files in os.walk(dicom_directory):
        dicom_files = [f for f in files if f.lower().endswith('.dcm')]
        mha_files = [f for f in files if f.lower().endswith('.mha')]
        
        for dicom_file, mha_file in zip(dicom_files, mha_files):
            dicom_path = os.path.join(subdir, dicom_file)
            mha_path = os.path.join(subdir, mha_file)
            features, path, image_filename, avg_pixel_int, area, outlined_filename = apply_mask_and_compute_features(dicom_path, mha_path, output_dir, target_area)
            study_id = os.path.basename(os.path.dirname(subdir))
            features_list.append((features, path, image_filename, study_id, avg_pixel_int, area, outlined_filename))
    
    if not features_list:
        print("No images processed or no features extracted.")
        return None

    # Save the features to a CSV file
    csv_filename = os.path.join(output_dir, f'{group_name}_features.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Study ID', 'Group', 'Image', 'Average Pixel Int', 'Homogeneity', 'Dissimilarity', 'Contrast', 'Correlation', 'Energy', 'Area', 'Outlined Image'])
        for features, path, image_filename, study_id, avg_pixel_int, area, outlined_filename in features_list:
            row = [
                study_id,
                group_name,
                image_filename,
                avg_pixel_int,
                features['homogeneity'],
                features['dissimilarity'],
                features['contrast'],
                features['correlation'],
                features['energy'],
                area,
                outlined_filename
            ]
            writer.writerow(row)

    return features_list


# Example usage
control = 'Analysis\\Control_Patients_Segmented'
fgr = 'Analysis\\FGR_Patients_Segmented'

control_features_list = process_directory_for_glcm(control, 'Control', target_area=30000)
fgr_features_list = process_directory_for_glcm(fgr, 'FGR', target_area=30000)
