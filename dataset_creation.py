import os
import numpy as np
import tensorflow as tf
import re

def extract_unique_ids(image_paths, class_name):
    id_set = set()
    regex = rf"{class_name}_([a-zA-Z0-9]+)"
    for path in image_paths:
        match = re.search(regex, path)
        if match:
            id_set.add(match.group(1))
    return len(id_set)

def load_dataset(data_dir, class_names, phase):
    image_paths, labels = [], []
    class_counts, unique_counts = {name: 0 for name in class_names}, {name: 0 for name in class_names}

    for i, name in enumerate(class_names):
        dir_path = os.path.join(data_dir, SUBDIRECTORIES[name][phase])
        files = os.listdir(dir_path)
        paths = [os.path.join(dir_path, file) for file in files]
        image_paths.extend(paths)
        labels.extend([i] * len(files))
        class_counts[name] += len(files)
        unique_counts[name] += extract_unique_ids(paths, name)

    print(f"{phase.capitalize()} set counts:")
    for name in class_names:
        print(f"{name}: {class_counts[name]}, Unique: {unique_counts[name]}")

    return np.array(image_paths), np.array(labels), unique_counts

def preprocess_and_resize_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = (image - tf.math.reduce_min(image)) * (255 / (tf.math.reduce_max(image) - tf.math.reduce_min(image)))
    image = tf.clip_by_value(image, 0, 255)
    return image, label

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image

def augment_and_preprocess_image(image, label):
    return tf.py_function(augment_image, [image], tf.float32), label

def create_datasets(data_dir, class_names, batch_size):
    train_images, train_labels, _ = load_dataset(data_dir, class_names, 'train')
    val_images, val_labels, _ = load_dataset(data_dir, class_names, 'val')

    train_labels = tf.keras.utils.to_categorical(train_labels, len(class_names))
    val_labels = tf.keras.utils.to_categorical(val_labels, len(class_names))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images))
    train_dataset = train_dataset.map(preprocess_and_resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(augment_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.map(preprocess_and_resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset
