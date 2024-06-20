import re
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def get_patient_identifier(image_path, class_name):
    if class_name == 'HCC':
        patient_id = re.findall('HCC_(.*?)-', image_path.split('/')[-1])[0]
    elif class_name == 'ICC':
        patient_id = re.findall('ICC_(.*?)_', image_path.split('/')[-1])[0]
    else:
        patient_id = re.findall('MCRC_mcrc_(.*?)_', image_path.split('/')[-1])[0]
    return patient_id

def prepare_patient_data(image_paths, labels, classes):
    patient_dict = {}
    for image_path, label, class_name in zip(image_paths, labels, classes):
        patient_id = get_patient_identifier(image_path, class_name)
        if patient_id not in patient_dict:
            patient_dict[patient_id] = {'image_paths': [], 'labels': [], 'classes': []}
        patient_dict[patient_id]['image_paths'].append(image_path)
        patient_dict[patient_id]['labels'].append(label)
        patient_dict[patient_id]['classes'].append(class_name)
    return patient_dict

def split_data(patient_dict, test_size=0.2, random_state=42):
    patient_ids = list(patient_dict.keys())
    train_ids, val_ids = train_test_split(patient_ids, test_size=test_size, random_state=random_state)

    train_paths, val_paths = [], []
    train_labels, val_labels = []

    for patient_id in train_ids:
        train_paths.extend(patient_dict[patient_id]['image_paths'])
        train_labels.extend(patient_dict[patient_id]['labels'])

    for patient_id in val_ids:
        val_paths.extend(patient_dict[patient_id]['image_paths'])
        val_labels.extend(patient_dict[patient_id]['labels'])

    return train_paths, train_labels, val_paths, val_labels

def oversample_training_data(train_paths, train_labels, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    paths, labels = ros.fit_resample(np.array(train_paths).reshape(-1, 1), train_labels)
    return paths.ravel(), labels

def count_unique_patients(patient_dict, val_ids, class_names):
    unique_patients = {class_name: set() for class_name in class_names}
    for patient_id in val_ids:
        for class_name in patient_dict[patient_id]['classes']:
            unique_patients[class_name].add(patient_id)
    return {class_name: len(patients) for class_name, patients in unique_patients.items()}

def count_images_per_class(image_paths):
    image_counts = {class_name: 0 for class_name in class_names}
    for path in image_paths:
        class_name = os.path.basename(os.path.dirname(path))
        image_counts[class_name] += 1
    return image_counts

def visualize_dataset_samples(train_dataset, class_names, samples_per_class=5):
    class_images = {class_name: [] for class_name in class_names}

    for images, labels in train_dataset.take(samples_per_class * len(class_names)):
        for image, label in zip(images, labels):
            class_idx = tf.argmax(label).numpy()
            class_name = class_names[class_idx]
            if len(class_images[class_name]) < samples_per_class:
                class_images[class_name].append(image)

    plt.figure(figsize=(10, 10))
    for i, class_name in enumerate(class_names):
        for j in range(samples_per_class):
            plt.subplot(len(class_names), samples_per_class, i * samples_per_class + j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(class_images[class_name][j])
            if j == 0:
                plt.title(class_name)
    plt.show()
