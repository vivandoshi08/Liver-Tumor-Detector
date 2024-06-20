from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3, Xception, ResNet152V2
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow as tf
import os
from sklearn.metrics import f1_score, recall_score, precision_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class FineTuneMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.best_val_accuracy = 0.0
        self.best_metrics = None

    def on_epoch_end(self, epoch, logs=None):
        val_labels, val_predictions = [], []

        for x, y in val_dataset:
            predictions = self.model.predict(x)
            val_predictions.extend(predictions.argmax(axis=1))
            val_labels.extend(y.numpy().argmax(axis=1))

        val_accuracy = logs['val_accuracy']
        f1 = f1_score(val_labels, val_predictions, average='macro')
        recall = recall_score(val_labels, val_predictions, average='macro')
        precision = precision_score(val_labels, val_predictions, average='macro')

        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.best_metrics = {'f1': f1, 'precision': precision, 'recall': recall}

    def on_train_end(self, logs=None):
        print(f"\nBest results for {self.model_name}:")
        print(f"  - Validation accuracy: {self.best_val_accuracy}")
        print(f"  - F1 score: {self.best_metrics['f1']}")
        print(f"  - Precision: {self.best_metrics['precision']}")
        print(f"  - Recall: {self.best_metrics['recall']}")

def fine_tune_model(model_name, model_fn, encoder_path, train_dataset, val_dataset, epochs=100, patience=30):
    inputs = layers.Input(shape=(299, 299, 3))
    base_model = model_fn(include_top=False, input_tensor=inputs, weights="imagenet")

    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

    model = Model(inputs, outputs)
    model.load_weights(encoder_path, by_name=True)

    for layer in model.layers:
        if layer.name in ['dense', 'dense_1', 'dense_2']:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

    checkpoint_path = f"/home/vivandoshi/Documents/liver_classifcation/{model_name}_fine_tuned.h5"
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
    early_stopping_callback = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, verbose=1)
    metrics_callback = FineTuneMetricsCallback(model_name)

    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=2, callbacks=[checkpoint_callback, early_stopping_callback, metrics_callback])
    return history, model

def fine_tune_all_models(models, losses, encoder_base_path, train_dataset, val_dataset, epochs=100, patience=30):
    best_model, best_history, best_accuracy = None, None, 0.0

    for model_name, model_fn in models.items():
        for loss in losses:
            encoder_path = os.path.join(encoder_base_path, f'{model_name}_encoder_{loss}.h5')
            history, model = fine_tune_model(model_name, model_fn, encoder_path, train_dataset, val_dataset, epochs, patience)

            if history.history['val_accuracy'][-1] > best_accuracy:
                best_accuracy = history.history['val_accuracy'][-1]
                best_model = model
                best_history = history

    return best_model, best_history, best_accuracy
