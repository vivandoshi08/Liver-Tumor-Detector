from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import InceptionV3, Xception, ResNet152V2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score, recall_score, precision_score

class CustomMetricsCallback(tf.keras.callbacks.Callback):
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

def build_and_train_models(train_dataset, val_dataset, models, epochs=30, img_size=299):
    callbacks_dict = {}
    for model_name, model_fn in models.items():
        print(f'Training {model_name}...')
        checkpoint_path = f"/home/vivandoshi/Documents/liver_classification/{model_name}_baseline.h5"

        inputs = layers.Input(shape=(img_size, img_size, 3))
        base_model = model_fn(include_top=False, input_tensor=inputs, weights="imagenet")

        for layer in base_model.layers:
            layer.trainable = False

        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

        model = Model(inputs, outputs)

        for layer in model.layers:
            if isinstance(layer, layers.Dense):
                layer.trainable = True
            else:
                layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
        early_stopping_callback = EarlyStopping(monitor="val_accuracy", mode="max", patience=10, verbose=1)
        metrics_callback = CustomMetricsCallback(model_name)
        callbacks_dict[model_name] = metrics_callback

        model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=2, callbacks=[checkpoint_callback, early_stopping_callback, metrics_callback])

    return callbacks_dict
