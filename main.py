from data_preprocessing import process_image_file
from dataset_creation import create_datasets
from model_training import build_and_train_models
from contrastive_learning import train_contrastive_models
from fine_tuning import fine_tune_all_models
from utils import prepare_patient_data, split_data, oversample_training_data, visualize_dataset_samples

# Define constants and paths
DATA_DIRECTORY = "/home/vivandoshi/Documents/liver_classifcation/Updated_Tumor_Slices"
CLASS_NAMES = ['HCC', 'ICC', 'MCRC']
BATCH_SIZE = 128
ENCODER_SAVE_DIRECTORY = "/home/vivandoshi/Documents/liver_classifcation/SimSiam_Pretrained"
ENCODER_DIRECTORY_PATH = "/home/vivandoshi/Documents/liver_classifcation/SimSiam_Pretrained/test"

# Create datasets
train_dataset, val_dataset = create_datasets(DATA_DIRECTORY, CLASS_NAMES, BATCH_SIZE)

# Define models to train
model_functions = {
    'XceptionModel': Xception,
    'InceptionV3Model': InceptionV3,
    'ResNet152V2Model': ResNet152V2
}

# Train baseline models
callbacks = build_and_train_models(train_dataset, val_dataset, model_functions)

# Train contrastive learning models
learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.0001, decay_steps=EPOCHS)
train_contrastive_models(['ResNet152V2Model'], ssl_dataset_train, learning_rate_schedule, ENCODER_SAVE_DIRECTORY)

# Fine-tune models
loss_functions = ['mse', 'cosine']
best_model, best_history, best_accuracy = fine_tune_all_models(model_functions, loss_functions, ENCODER_SAVE_DIRECTORY, train_dataset, val_dataset)
