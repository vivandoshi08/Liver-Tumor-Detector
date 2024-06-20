import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from tensorflow.keras.applications import InceptionV3, Xception, ResNet152V2

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

IMAGE_SIZE = (299, 299)
PROJECT_DIM = 1024
LATENT_DIM = 2048
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 100

all_image_paths = np.concatenate((train_image_paths, val_image_paths))
all_dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)

def custom_augmentation(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img

ssl_dataset_one = all_dataset.map(custom_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
ssl_dataset_two = all_dataset.map(custom_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

ssl_dataset_one = ssl_dataset_one.batch(BATCH_SIZE)
ssl_dataset_two = ssl_dataset_two.batch(BATCH_SIZE)

ssl_dataset_train = tf.data.Dataset.zip((ssl_dataset_one, ssl_dataset_two)).prefetch(tf.data.AUTOTUNE)

def build_encoder(model_name):
    if model_name == 'InceptionV3':
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    elif model_name == 'Xception':
        base_model = Xception(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    elif model_name == 'ResNet152V2':
        base_model = ResNet152V2(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    base_model.trainable = False

    inputs = base_model.input
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    return Model(inputs, x, name="encoder")

def build_predictor():
    return tf.keras.Sequential([
        layers.Input((128,)),
        layers.Dense(PROJECT_DIM, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
    ], name="predictor")

def calculate_cosine_loss(p, z):
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

def calculate_mse_loss(p, z):
    mse = MeanSquaredError()
    return mse(p, z)

class SimSiamModel(Model):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.cosine_loss_tracker = tf.keras.metrics.Mean(name="cosine_loss")
        self.mse_loss_tracker = tf.keras.metrics.Mean(name="mse_loss")

    @property
    def metrics(self):
        return [self.cosine_loss_tracker, self.mse_loss_tracker]

    def call(self, inputs):
        z = self.encoder(inputs)
        p = self.predictor(z)
        return p, z

    def train_step(self, data):
        ds_one, ds_two = data

        with tf.GradientTape() as tape:
            p1, z1 = self(ds_one)
            p2, z2 = self(ds_two)
            cosine_loss = (calculate_cosine_loss(p1, z2) + calculate_cosine_loss(p2, z1)) / 2
            mse_loss = (calculate_mse_loss(p1, z2) + calculate_mse_loss(p2, z1)) / 2
            total_loss = cosine_loss + mse_loss

        trainable_vars = self.encoder.trainable_variables + self.predictor.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.cosine_loss_tracker.update_state(cosine_loss)
        self.mse_loss_tracker.update_state(mse_loss)
        return {"cosine_loss": self.cosine_loss_tracker.result(), "mse_loss": self.mse_loss_tracker.result()}

def train_contrastive_models(base_models, ssl_dataset_train, lr_schedule, save_directory, patience=100):
    for model_name in base_models:
        encoder = build_encoder(model_name)
        predictor = build_predictor()
        simsiam_model = SimSiamModel(encoder, predictor)
        simsiam_model.build((None, *IMAGE_SIZE, 3))
        simsiam_model.compile(optimizer=Adam(lr_schedule))

        best_cosine_loss, best_mse_loss = float('inf'), float('inf')
        counter_cosine, counter_mse = 0, 0

        print(f"Training with {model_name}")

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            for data in ssl_dataset_train:
                results = simsiam_model.train_step(data)
                current_cosine_loss = results['cosine_loss'].numpy()
                current_mse_loss = results['mse_loss'].numpy()

                print(f"Cosine Loss: {current_cosine_loss}, MSE Loss: {current_mse_loss}")

                if current_cosine_loss < best_cosine_loss:
                    best_cosine_loss = current_cosine_loss
                    encoder.save(f'{save_directory}/{model_name}_encoder_cosine.h5')
                    counter_cosine = 0
                    print(f"Saved best cosine loss model with loss {best_cosine_loss}")
                else:
                    counter_cosine += 1

                if current_mse_loss < best_mse_loss:
                    best_mse_loss = current_mse_loss
                    encoder.save(f'{save_directory}/{model_name}_encoder_mse.h5')
                    counter_mse = 0
                    print(f"Saved best MSE loss model with loss {best_mse_loss}")
                else:
                    counter_mse += 1

                if counter_cosine > patience and counter_mse > patience:
                    print("Early stopping due to no improvement")
                    break

            simsiam_model.cosine_loss_tracker.reset_states()
            simsiam_model.mse_loss_tracker.reset_states()

            if counter_cosine > patience and counter_mse > patience:
                break
