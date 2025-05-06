import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, Dropout,
                                     GlobalMaxPooling1D, Lambda, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from skimage.transform import resize # Using scikit-image for resizing

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --- Configuration ---
# TODO: Set these paths according to your MRNet dataset location
BASE_DATA_DIR = './MRNet-v1.0/' # Base directory of the extracted MRNet dataset
OUTPUT_DIR = './output_models/' # Where to save trained models and logs

IMG_SIZE = (299, 299) # Input size for Xception
N_CHANNELS = 3 # Xception expects 3 channels
BATCH_SIZE = 8 # Adjust based on GPU memory. Smaller might be needed.
EPOCHS = 50 # Number of training epochs (can be adjusted with EarlyStopping)
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5

# --- Helper Functions ---

def load_labels(label_dir, task, split):
    """Loads labels for a specific task (acl, meniscus) and split (train, valid)."""
    label_path = os.path.join(label_dir, f"{split}-{task}.csv")
    labels_df = pd.read_csv(label_path, header=None, names=['exam_id', 'label'], index_col='exam_id')
    # Ensure index is string type if exam IDs in filenames are strings without .npy
    # labels_df.index = labels_df.index.astype(str)
    return labels_df['label'].to_dict()

def preprocess_slice(slice_img, target_size):
    """Preprocesses a single 2D slice."""
    # 1. Resize
    slice_resized = resize(slice_img, target_size, anti_aliasing=True)

    # 2. Normalize (Example: Scale to [0, 1]) - MRNet paper used scan-specific Z-score
    slice_normalized = (slice_resized - np.min(slice_resized)) / (np.max(slice_resized) - np.min(slice_resized) + 1e-6) # Add epsilon for stability

    # 3. Stack to 3 Channels
    slice_3channel = np.stack([slice_normalized] * 3, axis=-1)
    return slice_3channel.astype(np.float32)

# --- Keras Sequence for Data Loading ---

class MRNetSequence(Sequence):
    """ Keras Sequence for loading MRNet data slice by slice.
        Yields batches of (slices, labels_repeated_per_slice).
    """
    def __init__(self, data_dir, plane, labels_acl, labels_meniscus, exam_ids, batch_size, target_size):
        self.data_dir = data_dir
        self.plane = plane
        self.labels_acl = labels_acl
        self.labels_meniscus = labels_meniscus
        self.exam_ids = exam_ids # List of exam IDs for this sequence (train or valid)
        self.batch_size = batch_size # Note: this is exam batch size, actual slice batch size varies
        self.target_size = target_size
        self.indices = np.arange(len(self.exam_ids))
        # Do not shuffle here if validation; shuffle in on_epoch_end for training
        # np.random.shuffle(self.indices)

    def __len__(self):
        # Number of batches per epoch (based on exams)
        return int(np.ceil(len(self.exam_ids) / self.batch_size))

    def __getitem__(self, index):
        # Get batch indices for exams
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_exam_ids = [self.exam_ids[i] for i in batch_indices]

        batch_slices = []
        batch_labels_acl = []
        batch_labels_meniscus = []

        for exam_id in batch_exam_ids:
            # Load the 3D volume for the exam
            exam_path = os.path.join(self.data_dir, self.plane, f"{exam_id}.npy")
            try:
                volume = np.load(exam_path) # Shape: (num_slices, height, width)
            except FileNotFoundError:
                print(f"Warning: File not found {exam_path}. Skipping exam {exam_id}.")
                continue

            # Get exam labels
            label_acl = self.labels_acl.get(exam_id, None)
            label_meniscus = self.labels_meniscus.get(exam_id, None)

            if label_acl is None or label_meniscus is None:
                print(f"Warning: Labels not found for exam {exam_id}. Skipping.")
                continue

            # Preprocess each slice and collect
            for i in range(volume.shape[0]):
                slice_img = volume[i]
                processed_slice = preprocess_slice(slice_img, self.target_size)
                batch_slices.append(processed_slice)
                batch_labels_acl.append(label_acl)
                batch_labels_meniscus.append(label_meniscus)

        # Convert to numpy arrays
        batch_slices_np = np.array(batch_slices)
        batch_labels_acl_np = np.array(batch_labels_acl, dtype=np.float32)
        batch_labels_meniscus_np = np.array(batch_labels_meniscus, dtype=np.float32)

        # Return batch of slices and corresponding (repeated) exam labels
        # Keras expects labels in a format matching model output names or structure
        return batch_slices_np, {'acl_output': batch_labels_acl_np, 'meniscus_output': batch_labels_meniscus_np}

    def on_epoch_end(self):
        # Shuffle indices after each epoch for training data
        np.random.shuffle(self.indices)

# --- Build Model ---

def build_xception_model(input_shape, num_classes_acl=1, num_classes_meniscus=1, dropout_rate=0.5):
    """Builds the Xception model for slice-level prediction."""
    # Base Model (Xception)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape, pooling=None) # Pooling later
    base_model.trainable = False # Start with frozen base model

    # Input Layer
    slice_input = Input(shape=input_shape, name="slice_input")

    # Pass input through base model
    x = base_model(slice_input, training=False) # training=False important when layers are frozen

    # Pooling Layer (per slice)
    x = GlobalAveragePooling2D(name="slice_gap")(x)

    # Classification Heads
    x = Dropout(dropout_rate)(x)

    # Output layer for ACL
    acl_output = Dense(num_classes_acl, activation='sigmoid', name='acl_output')(x)

    # Output layer for Meniscus (sharing the GAP features)
    meniscus_output = Dense(num_classes_meniscus, activation='sigmoid', name='meniscus_output')(x)

    # Create Model
    model = Model(inputs=slice_input, outputs=[acl_output, meniscus_output], name="Xception_Slice_Classifier")

    return model

# --- Evaluation Function (with Aggregation) ---

def evaluate_model(model, data_dir, plane, labels_acl, labels_meniscus, exam_ids, target_size):
    """Evaluates the model on exam-level by aggregating slice predictions."""
    true_labels_acl = []
    true_labels_meniscus = []
    pred_probs_acl = []
    pred_probs_meniscus = []

    print(f"\nEvaluating on {len(exam_ids)} exams...")
    for exam_id in exam_ids:
        exam_path = os.path.join(data_dir, plane, f"{exam_id}.npy")
        try:
            volume = np.load(exam_path)
        except FileNotFoundError:
            print(f"Warning: Eval File not found {exam_path}. Skipping exam {exam_id}.")
            continue

        label_acl = labels_acl.get(exam_id, None)
        label_meniscus = labels_meniscus.get(exam_id, None)

        if label_acl is None or label_meniscus is None:
            print(f"Warning: Eval Labels not found for exam {exam_id}. Skipping.")
            continue

        exam_slices = []
        for i in range(volume.shape[0]):
            processed_slice = preprocess_slice(volume[i], target_size)
            exam_slices.append(processed_slice)

        if not exam_slices:
            print(f"Warning: No valid slices found for exam {exam_id}. Skipping.")
            continue

        exam_slices_np = np.array(exam_slices)

        # Get slice-level predictions
        slice_preds = model.predict(exam_slices_np, verbose=0) # Output is list [preds_acl, preds_meniscus]
        slice_preds_acl = slice_preds[0].flatten()
        slice_preds_meniscus = slice_preds[1].flatten()

        # Aggregate using Max Pooling
        exam_pred_acl = np.max(slice_preds_acl)
        exam_pred_meniscus = np.max(slice_preds_meniscus)

        # Store results
        true_labels_acl.append(label_acl)
        true_labels_meniscus.append(label_meniscus)
        pred_probs_acl.append(exam_pred_acl)
        pred_probs_meniscus.append(exam_pred_meniscus)

    # Calculate Metrics
    print("\n--- Evaluation Results ---")
    if not true_labels_acl: # Check if any exams were processed
         print("No valid exams found for evaluation.")
         return None

    true_labels_acl = np.array(true_labels_acl)
    true_labels_meniscus = np.array(true_labels_meniscus)
    pred_probs_acl = np.array(pred_probs_acl)
    pred_probs_meniscus = np.array(pred_probs_meniscus)
    pred_labels_acl = (pred_probs_acl > 0.5).astype(int)
    pred_labels_meniscus = (pred_probs_meniscus > 0.5).astype(int)

    auc_acl = roc_auc_score(true_labels_acl, pred_probs_acl)
    acc_acl = accuracy_score(true_labels_acl, pred_labels_acl)
    loss_acl = log_loss(true_labels_acl, pred_probs_acl)

    auc_meniscus = roc_auc_score(true_labels_meniscus, pred_probs_meniscus)
    acc_meniscus = accuracy_score(true_labels_meniscus, pred_labels_meniscus)
    loss_meniscus = log_loss(true_labels_meniscus, pred_probs_meniscus)

    avg_auc = (auc_acl + auc_meniscus) / 2.0

    print(f"ACL Task:")
    print(f"  AUC: {auc_acl:.4f}")
    print(f"  Accuracy: {acc_acl:.4f}")
    print(f"  Log Loss: {loss_acl:.4f}")
    print(f"Meniscus Task:")
    print(f"  AUC: {auc_meniscus:.4f}")
    print(f"  Accuracy: {acc_meniscus:.4f}")
    print(f"  Log Loss: {loss_meniscus:.4f}")
    print(f"Average Exam AUC: {avg_auc:.4f}")
    print("------------------------")

    return avg_auc # Return average AUC for potential use in validation callback

# --- Main Training Script ---

def main(args):
    # Set GPU Memory Growth (prevents TensorFlow from allocating all GPU memory at once)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e) # Memory growth must be set before GPUs have been initialized

    # --- Setup ---
    print(f"Selected Plane: {args.plane}")
    train_data_dir = os.path.join(args.base_dir, 'train')
    valid_data_dir = os.path.join(args.base_dir, 'valid')
    label_dir = args.base_dir # Assuming label CSVs are in the base directory

    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, f'xception_mrnet_{args.plane}.h5')

    # --- Load Labels ---
    print("Loading labels...")
    train_labels_acl = load_labels(label_dir, 'acl', 'train')
    train_labels_meniscus = load_labels(label_dir, 'meniscus', 'train')
    valid_labels_acl = load_labels(label_dir, 'acl', 'valid')
    valid_labels_meniscus = load_labels(label_dir, 'meniscus', 'valid')

    # Get list of exam IDs for train/valid splits (assuming IDs are filenames without .npy)
    train_exam_ids = sorted([int(f.split('.')[0]) for f in os.listdir(os.path.join(train_data_dir, args.plane)) if f.endswith('.npy')])
    valid_exam_ids = sorted([int(f.split('.')[0]) for f in os.listdir(os.path.join(valid_data_dir, args.plane)) if f.endswith('.npy')])
    print(f"Found {len(train_exam_ids)} training exams and {len(valid_exam_ids)} validation exams for plane {args.plane}.")

    # --- Create Data Generators ---
    print("Creating data generators...")
    train_gen = MRNetSequence(train_data_dir, args.plane, train_labels_acl, train_labels_meniscus,
                              train_exam_ids, args.batch_size, IMG_SIZE)
    valid_gen = MRNetSequence(valid_data_dir, args.plane, valid_labels_acl, valid_labels_meniscus,
                              valid_exam_ids, args.batch_size, IMG_SIZE) # Use same batch size for consistency, or smaller if memory limited

    # --- Build and Compile Model ---
    print("Building model...")
    model = build_xception_model(input_shape=(*IMG_SIZE, N_CHANNELS),
                                 dropout_rate=args.dropout_rate)

    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss={'acl_output': BinaryCrossentropy(), 'meniscus_output': BinaryCrossentropy()},
                  loss_weights={'acl_output': 1.0, 'meniscus_output': 1.0}, # Equal weighting
                  metrics={'acl_output': [AUC(name='auc_acl'), BinaryAccuracy(name='acc_acl')],
                           'meniscus_output': [AUC(name='auc_meniscus'), BinaryAccuracy(name='acc_meniscus')]})

    print(model.summary())

    # --- Callbacks ---
    checkpoint = ModelCheckpoint(model_save_path,
                                 monitor='val_loss', # Monitor validation loss (sum of both tasks)
                                 save_best_only=True,
                                 save_weights_only=False, # Save entire model
                                 mode='min',
                                 verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10, # Stop after 10 epochs with no improvement
                                   mode='min',
                                   restore_best_weights=True, # Restore weights from best epoch
                                   verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2, # Reduce LR by factor of 5
                                  patience=5,
                                  mode='min',
                                  min_lr=1e-6,
                                  verbose=1)

    # --- Train Model ---
    print("Starting training...")
    history = model.fit(train_gen,
                        epochs=args.epochs,
                        validation_data=valid_gen,
                        callbacks=[checkpoint, early_stopping, reduce_lr],
                        verbose=1) # Set verbose=1 or 2 for progress

    print("Training finished.")

    # --- Final Evaluation ---
    # Load the best model saved by ModelCheckpoint
    print(f"Loading best model from {model_save_path} for final evaluation...")
    best_model = tf.keras.models.load_model(model_save_path)

    print("\n--- Final Evaluation on Validation Set ---")
    evaluate_model(best_model, valid_data_dir, args.plane, valid_labels_acl, valid_labels_meniscus,
                   valid_exam_ids, IMG_SIZE)

    # --- Optional: Fine-tuning ---
    # If you want to fine-tune:
    # 1. Unfreeze some layers of the base model
    #    base_model = best_model.layers[1] # Get the base Xception layer
    #    base_model.trainable = True
    #    # Fine-tune only the top layers (e.g., last few blocks)
    #    for layer in base_model.layers[:-30]: # Example: freeze all but last 30 layers
    #        layer.trainable = False
    # 2. Re-compile with a very low learning rate
    #    best_model.compile(...) # Use Adam(learning_rate=1e-5 or lower)
    # 3. Continue training for a few more epochs
    #    history_fine_tune = best_model.fit(...)

    print(f"Model for plane {args.plane} saved to {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Xception on MRNet dataset.')
    parser.add_argument('--base_dir', type=str, default='./MRNet-v1.0/', help='Path to the base MRNet directory.')
    parser.add_argument('--output_dir', type=str, default='./output_models/', help='Directory to save trained models.')
    parser.add_argument('--plane', type=str, required=True, choices=['axial', 'coronal', 'sagittal'], help='Which MRI plane to train on.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size (number of exams per batch).')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE, help='Dropout rate.')

    args = parser.parse_args()

    # Override constants with parsed args if provided
    BASE_DATA_DIR = args.base_dir
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    DROPOUT_RATE = args.dropout

    main(args)