import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage.measure import label, regionprops
from tqdm import tqdm
import cv2
import re
import heapq

# Directory paths
IMAGE_DIRECTORIES = {
    'HCC': '/home/vivandoshi/Documents/liver_classifcation/images/HCC',
    'ICC': '/home/vivandoshi/Documents/liver_classifcation/images/ICC',
    'MCRC': '/home/vivandoshi/Documents/liver_classifcation/images/MCRC'
}

LABEL_DIRECTORIES = {
    'HCC': '/home/vivandoshi/Documents/liver_classifcation/labels/HCC',
    'ICC': '/home/vivandoshi/Documents/liver_classifcation/labels/ICC',
    'MCRC': '/home/vivandoshi/Documents/liver_classifcation/labels/MCRC'
}

OUTPUT_DIRECTORY = '/home/vivandoshi/Documents/liver_classifcation/Processed_Images'

SUBDIRECTORIES = {
    'HCC': {'train': os.path.join(OUTPUT_DIRECTORY, 'HCC', 'train'), 'val': os.path.join(OUTPUT_DIRECTORY, 'HCC', 'val')},
    'ICC': {'train': os.path.join(OUTPUT_DIRECTORY, 'ICC', 'train'), 'val': os.path.join(OUTPUT_DIRECTORY, 'ICC', 'val')},
    'MCRC': {'train': os.path.join(OUTPUT_DIRECTORY, 'MCRC', 'train'), 'val': os.path.join(OUTPUT_DIRECTORY, 'MCRC', 'val')}
}

SLICE_DIMS = {'HCC': (180, 233), 'ICC': (180, 233), 'MCRC': (180, 233)}

def adjust_window(ct_slice, window_level, window_width):
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    mask = ct_slice != 0
    ct_slice[mask] = np.clip(ct_slice[mask], lower_bound, upper_bound)
    ct_slice[mask] = (ct_slice[mask] - lower_bound) / (upper_bound - lower_bound)
    return ct_slice

def extract_largest_tumor_slices(label_array, num_slices=15, min_area=50):
    labeled_image = label(label_array == 2)
    regions = regionprops(labeled_image)
    slice_areas = [(z, np.sum(label_array[:, :, z] == 2)) for z in range(label_array.shape[2]) if np.sum(label_array[:, :, z] == 2) >= min_area]
    largest_slices = heapq.nlargest(num_slices, slice_areas, key=lambda x: x[1])
    return [z for z, area in largest_slices]

def crop_tumor_image(ct_slice, label_slice, tumor_type):
    tumor_mask = label_slice == 2
    indices = np.where(tumor_mask)
    if len(indices[0]) == 0:
        return np.zeros_like(ct_slice), np.zeros_like(label_slice)

    center = np.mean(indices, axis=1)
    half_dims = np.array(SLICE_DIMS[tumor_type]) / 2
    min_idx = np.round(center - half_dims).astype(int)
    max_idx = np.round(center + half_dims).astype(int)

    cropped_ct = ct_slice[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1]]
    cropped_label = label_slice[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1]]
    return cropped_ct, cropped_label

def process_tumor_slices(ct_data, label_data, slices, tumor_type, filename, output_directory):
    for slice_idx in slices:
        ct_slice = ct_data[:, :, slice_idx]
        label_slice = label_data[:, :, slice_idx]
        cropped_ct, cropped_label = crop_tumor_image(ct_slice, label_slice, tumor_type)
        ct_tumor_only = np.where(cropped_label == 2, cropped_ct, 0)
        adjusted_ct = adjust_window(ct_tumor_only, 40, 350)

        if np.any(adjusted_ct):
            resized_ct = resize(adjusted_ct, (299, 299), anti_aliasing=True)
            rotated_ct = np.rot90(resized_ct, k=-1)
            rgb_ct = np.stack((rotated_ct,) * 3, axis=-1)
            output_filename = f"{tumor_type}_{filename}_slice{slice_idx}.png"
            output_path = os.path.join(output_directory, output_filename)
            cv2.imwrite(output_path, (rgb_ct * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            print(f"Skipping empty image {filename}.")

def process_image_file(image_file, label_file, tumor_type, output_directory):
    ct_data = nib.load(image_file).get_fdata()
    label_data = nib.load(label_file).get_fdata()
    top_slices = extract_largest_tumor_slices(label_data)
    filename = os.path.splitext(os.path.basename(image_file))[0]
    process_tumor_slices(ct_data, label_data, top_slices, tumor_type, filename, output_directory)

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

for tumor_type in IMAGE_DIRECTORIES.keys():
    image_path = IMAGE_DIRECTORIES[tumor_type]
    label_path = LABEL_DIRECTORIES[tumor_type]
    image_files = sorted(os.listdir(image_path), key=lambda x: int(re.findall(r'\d+', x)[0]))
    label_files = sorted(os.listdir(label_path), key=lambda x: int(re.findall(r'\d+', x)[0]))
    image_label_pairs = [(os.path.join(image_path, img), os.path.join(label_path, lbl)) for img, lbl in zip(image_files, label_files)]
    train_images, val_images = train_test_split(image_label_pairs, test_size=0.2, random_state=42)

    for split, images in zip(['train', 'val'], [train_images, val_images]):
        split_dir = SUBDIRECTORIES[tumor_type][split]
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for image_file, label_file in tqdm(images, desc=f"{tumor_type} - {split.capitalize()}"):
            process_image_file(image_file, label_file, tumor_type, split_dir)
