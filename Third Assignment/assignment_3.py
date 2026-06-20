
import os
import pickle
import datetime
import splitfolders
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight



def resize_dataset_images(source_dir: str, dest_dir: str, image_shape: tuple):
    """
        Given a dataset, resize all images using opencv.
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        for class_name in os.listdir(source_dir):
            class_source_path = os.path.join(source_dir, class_name)

            if not os.path.isdir(class_source_path):
                continue

            class_dest_path = os.path.join(dest_dir, class_name)
            os.makedirs(class_dest_path, exist_ok=True)

            for filename in os.listdir(class_source_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_source_path, filename)
                    img_bgr = cv2.imread(img_path)
                    img_resized = cv2.resize(img_bgr, image_shape, interpolation=cv2.INTER_LINEAR)
                    new_img_path = os.path.join(class_dest_path, filename)
                    cv2.imwrite(new_img_path, img_resized)


def convert_tif_to_png(source_dir: str, dest_dir: str):
    """
        Converts UCMerced_LandUse images from tif to png.
    """
    if os.path.exists(dest_dir):
        return

    os.makedirs(dest_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_source_path = os.path.join(source_dir, class_name)

        if not os.path.isdir(class_source_path):
            continue

        class_dest_path = os.path.join(dest_dir, class_name)
        os.makedirs(class_dest_path, exist_ok=True)

        for filename in os.listdir(class_source_path):
            if filename.lower().endswith(".tif"):
                img_path = os.path.join(class_source_path, filename)
                new_filename = os.path.splitext(filename)[0] + ".png"
                new_img_path = os.path.join(class_dest_path, new_filename)
                img = cv2.imread(img_path)
                cv2.imwrite(new_img_path, img)


def setup_gpu():
    """
       Let TensorFlow use available GPUs, otherwise CPU.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return "CPU"

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    return "GPU"


def calculate_class_weights(dataset_dir: str, dir_to_save: str, class_names: list) -> dict:
    """
        Handle unbalanced dataset by computing weights for each class.
        These weights are used during the training when computing the loss.

        Return a dictionary -> {CLASS_INDEX: CLASS_WEIGHT, ...}
    """

    class_weights_file = os.path.join(dir_to_save, f"{os.path.basename(dataset_dir)}_weights_for_each_class.pkl")
    if not os.path.exists(class_weights_file):
        class_image_counts = {}
        for class_name in os.listdir(dataset_dir):
            class_path = os.path.join(dataset_dir, class_name)
            image_count = sum(1 for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            class_image_counts[class_name] = image_count

        # Creo un vettore di frequenze assolute: l'indice relativo ad una classe
        # si ripete un numero di volte pari alla numerosità della classe stessa
        counts = np.array([class_image_counts[name] for name in class_names])
        y = np.repeat(np.arange(len(class_names)), counts)

        # peso_classe_i = N_totale / (N_classi x N_immagini_classe_i)
        # Creo un dizionario -> {0: peso_0, 1: peso_1, ...}
        class_weights_dict = dict(enumerate(compute_class_weight(class_weight='balanced', classes=np.arange(len(class_names)), y=y)))
        pickle.dump(class_weights_dict, open(class_weights_file, "wb"))

    class_weights_dict = pickle.load(open(class_weights_file, "rb"))

    return class_weights_dict


def train_val_test_split(input_folder: str, output_folder: str, ratio: tuple):
    """
        Make a stratified splitting. Creating copies on the filestystem allow us to
        benefit from TensorFlow load dataset in the next steps.
    """

    if not os.path.exists(output_folder):
        splitfolders.ratio(input_folder, output=output_folder,
                           seed=177, ratio=ratio,
                           group_prefix=None, move=False)  # move=True li sposta


def load_train_val_sets(dataset_dir: str) -> tuple:
    """
    Let TensorFlow loads train and validation sets build previously.

    We also specify the size of the batch.

    At the end there are some optimizations:
        - While GPU is working, CPU prepares the loading of the next batch (prefetch).

        - The train set is shuffled at each epoch: images in each batch change during epochs.

        - Data augmentation is applied, in particular only to the train set.

        - Validation set is not shuffled,
          in this way we guarantee consistency of the calculated metrics.
    """

    batch_size = 16

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_dir, "train"),
        seed=177,
        shuffle=True,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='categorical'
    )

    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_dir, "val"),
        seed=177,
        shuffle=False,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='categorical'
    )

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
    ], name="data_augmentation_pipeline")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def convolutional_backbone(img_shape:tuple = (224, 224, 3)):
    """
        The Backbone takes a 224x224x3 image and outputs a 6272‑D feature vector (obtained by flattening 7x7x128 feature maps).
        The idea is to use a first stage of filters to extract features from the image and then refine them by using
        a sequence of residual blocks.

        The backbone is composed as follows:

        - A rescaling step that rescales pixel values to the [0, 1] range.

        - A first convolutional stage which reduces the spatial resolution.
          This strategy is often called the stem of the network.
          It takes as input feature maps of 224x224x3 and returns feature maps of 56x56x64.

        - Three residual blocks. The first residual block starts from 56x56x64 and the sequence of three blocks
          progressively produces 7x7x256 feature maps.
          Each block follows a common structure, here an example:
                anchor node -> conv -> conv -> conv -> add skip connection with anchor -> activation
                When the number of channels changes, the skip connection is projected with a 1x1 convolution
                so that the addition is valid.

          Furthermore, at the end of each residual block there is a MaxPooling. In this way, we reduce the dimension
          of each feature map. This is helpful not only to focus on macro details, but also to have a smaller
          feature vector for the input of the MLP.

        - The last convolutional stage is a 1x1 convolution with 128 filters.
          In this way, from 7x7x256, we produce feature maps of 7x7x128.

        - At the end there is a flatten layer to transform the final feature maps into a 1D feature vector of 6272‑D.
    """

    # La dimensione è relativa a una immagine RGB
    inputs = layers.Input(shape=img_shape)

    x = layers.Rescaling(scale=1. / 255)(inputs)

    # Il bias viene annullato dalla normalizzazione, quindi è inutile calcolarlo
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    anchor_node_1 = x

    # Primo blocco residuale #
    x = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, anchor_node_1])
    x = layers.Activation('relu', name="output_first_residual_block")(x)

    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    anchor_node_2 = x

    # Secondo blocco residuale #
    x = layers.Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    shortcut_2 = layers.Conv2D(128, kernel_size=1, padding='same', use_bias=False)(anchor_node_2)
    shortcut_2 = layers.BatchNormalization()(shortcut_2)
    x = layers.Add()([x, shortcut_2])
    x = layers.Activation('relu', name="output_second_residual_block")(x)

    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    anchor_node_3 = x

    # Terzo blocco residuale #
    x = layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    shortcut_3 = layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(anchor_node_3)
    shortcut_3 = layers.BatchNormalization()(shortcut_3)
    x = layers.Add()([x, shortcut_3])
    x = layers.Activation('relu', name="output_third_residual_block")(x)

    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    # Passaggio da mappe 7x7x256 a 7x7x128
    x = layers.Conv2D(128, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    outputs = layers.Flatten(name="output_conv_blocks")(x)

    backbone = models.Model(inputs=inputs, outputs=outputs, name="ConvBackbone")

    return backbone


def GaidNet(img_shape:tuple = (224, 224, 3)):
    """
        GaidNet stands for Gabro AID Net.

        The MLP takes the 6272‑D backbone feature vector as input.

        We use a convolutional backbone linked to an MLP with a softmax at the end.
        The MLP is used for classification: it has three layer dense, and it outputs the result using a layer of 30 neurons.
        We use 30 neurons because this is the number of classes for the AID dataset.
        The output of this final layer uses a softmax activation to return the class probabilities.

        At the end we guarantee also a balance about the number of parameters for the backbone and the MLP:
        we have about 1 million parameters for each.
    """

    inputs = layers.Input(shape=img_shape)
    backbone = convolutional_backbone(img_shape)

    features = backbone(inputs)

    # MLP #
    x = layers.Dense(256, use_bias=False)(features)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(100, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name="features_before_classifier_and_dropout")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(30, activation='softmax', name="aid_classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="GaidNet")

    return model


def GumerNet():
    """
        GumerNet stands for Gabro UCMerced_LandUse Net.

        We load the best GaidNet checkpoint and freeze its convolutional
        backbone so that its weights are not updated during training.

        Then we replace the original 30-neuron classification head with a new layer dense of 21 neurons,
        which corresponds to the number of classes for the UCMerced_LandUse dataset.
        The output uses a softmax activation to return the class probabilities.
        This all new MLP is trainable.

        At the end we guarantee also a balance about the number of parameters for the backbone and the MLP:
        we have about 1 million parameters for each.
    """

    pretrain_dir = os.path.join(os.getcwd(), "checkpoints", "first_strategy", "pretrain")
    best_pretrained_path = os.path.join(pretrain_dir, "GaidNet_best.keras")

    base_model = load_model(best_pretrained_path)
    backbone_model = base_model.get_layer("ConvBackbone")
    backbone_model.trainable = False

    x = base_model.get_layer("features_before_classifier_and_dropout").output
    x = layers.Dropout(0.4)(x)

    ucmerced_outputs = layers.Dense(21, activation='softmax', name="ucmerced_classifier")(x)
    gumer_net = models.Model(inputs=base_model.input, outputs=ucmerced_outputs, name="GumerNet")

    return gumer_net


def ZeroGumerNet(img_shape:tuple = (224, 224, 3)):
    """
        ZeroGumerNet stands for Zero Gabro UCMerced_LandUse Net.

        The MLP takes the 6272‑D backbone feature vector as input.

        We use a convolutional backbone linked to an MLP with a softmax at the end.
        The MLP is used for classification: it has three layer dense, and it outputs the result using a layer of 21 neurons.
        We use 21 neurons because this is the number of classes for the UCMerced_LandUse dataset.
        The output of this final layer uses a softmax activation to return the class probabilities.

        At the end we guarantee also a balance about the number of parameters for the backbone and the MLP:
        we have about 1 million parameters for each.
    """

    inputs = layers.Input(shape=img_shape)
    backbone = convolutional_backbone(img_shape)

    features = backbone(inputs)

    # MLP #
    x = layers.Dense(256, use_bias=False)(features)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.6)(x)

    x = layers.Dense(100, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(21, activation='softmax', name="zero_ucmerced_classifier")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="ZeroGumerNet")

    return model


def get_callbacks(model_name: str, dir_tosave: str, es_patience: int,
                  lr_patience: int, lr_factor: float, min_lr: float) -> list:
    """
        Functions to call after each epoch:

        - EarlyStopping: stops training if the validation loss does not improve by at least `min_delta`
                         for `patience` consecutive epochs.
        - Reduce Learning Rate: decreases the learning rate by `factor` if the validation loss does not
                                improve by at least `min_delta` for `lr_patience` consecutive epochs,
                                down to a minimum of `min_lr`.
        - Model Checkpoint: save the model with the best validation loss. This guarantee the best model is always the one saved.
        - CSV Logger: save -> epoch, accuracy, loss, lr, val_accuracy, val_loss.
        - Tensorboard: powerful tool, saved just in case.

        Note that:
            In each training configuration, the patience of EarlyStopping is set to be greater than
            lr_patience to allow the learning rate to reduce (by lr_factor) at least 3 times before stopping,
            giving the model enough time to escape plateaus.
    """

    log_dir = os.path.join(dir_tosave, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_checkpoint_path = os.path.join(dir_tosave, f"{model_name}_best.keras")
    csv_logger_path = os.path.join(dir_tosave, f"{model_name}_training_history.csv")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=es_patience, restore_best_weights=False, min_delta=1e-4, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             factor=lr_factor, patience=lr_patience, min_lr=min_lr, min_delta=1e-4, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(csv_logger_path),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    return callbacks


def train_model(model, model_name: str, train_ds, class_weights: dict, val_ds, dir_tosave: str, epochs: int,
                lr: float, es_patience: int, lr_patience: int, lr_factor: float, min_lr: float):
    """
        Common training function used for all nets: GaidNet, GumerNet, ZeroGumerNet.

        Compiles the model with Adam optimizer and Categorical Cross-Entropy loss.
        Then trains it using the provided datasets.
        Class weights are applied to handle class imbalance.
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    callbacks = get_callbacks(model_name=model_name, dir_tosave=dir_tosave,
                              es_patience=es_patience, lr_patience=lr_patience,
                              lr_factor=lr_factor, min_lr=min_lr)

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, class_weight=class_weights, verbose=1)

    model_checkpoint_path = os.path.join(dir_tosave, f"{model_name}_best.keras")
    # Considera i pesi salvati grazie a ModelCheckpoint
    model.load_weights(model_checkpoint_path)
    return history


def plot_loss_and_accuracy(base_dir: str, model_name: str, history_csv_dir: str):
    """
        Make plots for train and validation set of:
        - Loss vs Epochs
        - Accuracy vs Epochs
    """

    evaluation_path = os.path.join(base_dir, "evaluation")
    os.makedirs(evaluation_path, exist_ok=True)

    evaluation_path = os.path.join(base_dir, "evaluation")
    os.makedirs(evaluation_path, exist_ok=True)

    history_df = pd.read_csv(history_csv_dir)
    epochs_range = history_df['epoch'] if 'epoch' in history_df.columns else history_df.index

    best_epoch_idx = history_df['val_loss'].idxmin()
    best_epoch = epochs_range[best_epoch_idx]
    best_val_accuracy = history_df.loc[best_epoch_idx, 'val_accuracy']

    x_offset = (epochs_range.max() - epochs_range.min()) * 0.01

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, history_df['loss'], label='Train', linewidth=2, color='#2563EB')
    ax.plot(epochs_range, history_df['val_loss'], label='Val', linewidth=2, color='#DC2626')
    ax.axvline(x=best_epoch, color='gray', linestyle='--')

    ax.text(best_epoch - 1.75, 0.98, f'Epoch {best_epoch}',
            transform=ax.get_xaxis_transform(),
            color='gray', ha='center', va='top', fontsize=9, fontweight='bold')

    ax.set_title(f"Loss vs Epochs ({model_name})", fontsize=12, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Categorical Crossentropy')
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(-0.02, 5.76)
    fig.savefig(os.path.join(evaluation_path, f"loss_curve_{model_name}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, history_df['accuracy'], label='Train', linewidth=2, color='#2563EB')
    ax.plot(epochs_range, history_df['val_accuracy'], label='Val', linewidth=2, color='#DC2626')
    ax.axvline(x=best_epoch, color='gray', linestyle='--')

    ax.text(best_epoch - 1.75, 0.98, f'Epoch {best_epoch}',
            transform=ax.get_xaxis_transform(),
            color='gray', ha='center', va='top', fontsize=9, fontweight='bold')

    ax.text(best_epoch + x_offset, best_val_accuracy - 0.05,
            f'Val acc: {best_val_accuracy:.3f}',
            color='gray', ha='left', va='bottom', fontsize=9)

    ax.set_title(f"Accuracy vs Epochs ({model_name})", fontsize=12, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=15))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.savefig(os.path.join(evaluation_path, f"accuracy_curve_{model_name}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def pre_training(train: bool = False, epochs: int=200, lr: float=1e-3, es_patience: int=20,
                  lr_patience: int=5, lr_factor: float=0.2, min_lr: float = 1e-6, make_plots: bool = True):
    """
        Support function used for the pretraining phase.

        Prepare train and validation sets, then load the model and start the training.
        At the end is possible to make plots of loss and accuracy vs epochs.
    """

    pretrain_dir = os.path.join(os.getcwd(), "checkpoints", "first_strategy", "pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)

    if train:
        # Load Residual Network and save summary
        model = GaidNet(img_shape=(224, 224, 3))
        model_summary_path = os.path.join(pretrain_dir, "model_summary.txt")
        with open(model_summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        # Prepare TRAIN and VAL
        original_dataset_dir = os.path.join(os.getcwd(), "DATASET", "AID")
        dataset_dir = os.path.join(os.getcwd(), "DATASET", "AID_resized")
        resize_dataset_images(original_dataset_dir, dataset_dir, (224, 224))
        stratified_dataset_dir = os.path.join(os.getcwd(), "DATASET", "AID_stratified")
        train_val_test_split(input_folder=dataset_dir, output_folder=stratified_dataset_dir, ratio=(0.7, 0.3))

        # Load TRAIN and VAL
        train_ds, val_ds, class_names = load_train_val_sets(stratified_dataset_dir)

        # Handle unbalanced classes (using only train set details)
        train_dir = os.path.join(stratified_dataset_dir, "train")
        class_weights = calculate_class_weights(dataset_dir=train_dir, dir_to_save=pretrain_dir, class_names=class_names)

        # START PRETRAINING
        history = train_model(model, model_name="GaidNet",
                              train_ds=train_ds, class_weights=class_weights,
                              val_ds=val_ds,
                              dir_tosave=pretrain_dir,
                              epochs=epochs, lr=lr,
                              es_patience=es_patience, lr_patience=lr_patience,
                              lr_factor=lr_factor, min_lr=min_lr)

    if make_plots:
        # Evaluate best pretrained model
        history_csv_dir = os.path.join(pretrain_dir, "GaidNet_training_history.csv")
        plot_loss_and_accuracy(base_dir=pretrain_dir, model_name="GaidNet_best.keras", history_csv_dir=history_csv_dir)


def finetuning_step(train: bool = False, epochs: int=1000, lr: float=1e-4, es_patience: int=20,
                    lr_patience: int=4, lr_factor: float=0.2, min_lr: float = 1e-7, make_plots: bool = True):
    """
        Support function used for the finetuning phase.

        Prepare train and validation sets, then load the model and start the training.
        At the end is possible to make plots of loss and accuracy vs epochs.
    """

    finetuning_dir = os.path.join(os.getcwd(), "checkpoints", "first_strategy", "finetuning")
    os.makedirs(finetuning_dir, exist_ok=True)

    if train:
        # Load Residual Network (freezed Backbone + trainable MLP with new head) and save summary
        model = GumerNet()
        model_summary_path = os.path.join(finetuning_dir, "model_summary.txt")
        with open(model_summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        # Prepare TRAIN and VAL
        original_dataset_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse")
        converted_dataset_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse_Converted")
        convert_tif_to_png(source_dir=original_dataset_dir, dest_dir=converted_dataset_dir)

        resized_dataset_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse_resized")
        resize_dataset_images(converted_dataset_dir, resized_dataset_dir, (224, 224))

        stratified_dataset_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse_stratified")
        train_val_test_split(input_folder=resized_dataset_dir, output_folder=stratified_dataset_dir, ratio=(0.7, 0.15, 0.15))

        # Load TRAIN and VAL
        train_ds, val_ds, class_names = load_train_val_sets(stratified_dataset_dir)

        # Handle unbalanced classes (using only train set details)
        train_dir = os.path.join(stratified_dataset_dir, "train")
        class_weights = calculate_class_weights(dataset_dir=train_dir, dir_to_save=finetuning_dir, class_names=class_names)

        # START FINETUNING
        history = train_model(model, model_name="GumerNet",
                              train_ds=train_ds, class_weights=class_weights,
                              val_ds=val_ds,
                              dir_tosave=finetuning_dir,
                              epochs=epochs, lr=lr,
                              es_patience=es_patience, lr_patience=lr_patience,
                              lr_factor=lr_factor, min_lr=min_lr)

    if make_plots:
        # Evaluate best finetuned model
        history_csv_dir = os.path.join(finetuning_dir, "GumerNet_training_history.csv")
        plot_loss_and_accuracy(base_dir=finetuning_dir, model_name="GumerNet_best.keras", history_csv_dir=history_csv_dir)


def second_strategy(train: bool = False, epochs: int=200, lr: float=1e-3, es_patience: int=16,
                    lr_patience: int=4, lr_factor: float=0.2, min_lr: float = 1e-6, make_plots: bool = True):
    """
        Support function used for the second strategy phase.

        Prepare train and validation sets, then load the model and start the training.
        At the end is possible to make plots of loss and accuracy vs epochs.
    """

    second_strategy_dir = os.path.join(os.getcwd(), "checkpoints", "second_strategy")
    os.makedirs(second_strategy_dir, exist_ok=True)

    if train:
        # Load new Residual Network and save summary
        model = ZeroGumerNet(img_shape=(224, 224, 3))
        model_summary_path = os.path.join(second_strategy_dir, "model_summary.txt")
        with open(model_summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        # Prepare TRAIN and VAL
        original_dataset_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse")
        converted_dataset_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse_Converted")
        convert_tif_to_png(source_dir=original_dataset_dir, dest_dir=converted_dataset_dir)

        resized_dataset_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse_resized")
        resize_dataset_images(converted_dataset_dir, resized_dataset_dir, (224, 224))

        stratified_dataset_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse_stratified")
        train_val_test_split(input_folder=resized_dataset_dir, output_folder=stratified_dataset_dir, ratio=(0.7, 0.15, 0.15))

        # Load TRAIN and VAL
        train_ds, val_ds, class_names = load_train_val_sets(stratified_dataset_dir)


        # Save VALIDATION and class names a part
        val_ds_path = os.path.join(second_strategy_dir, "val_ds")
        tf.data.Dataset.save(val_ds, val_ds_path)
        class_names_path = os.path.join(second_strategy_dir, "class_names.pkl")
        pickle.dump(class_names, open(class_names_path, "wb"))


        # Handle unbalanced classes (using only train set details)
        train_dir = os.path.join(stratified_dataset_dir, "train")
        class_weights = calculate_class_weights(dataset_dir=train_dir, dir_to_save=second_strategy_dir, class_names=class_names)

        # START TRAINING
        history = train_model(model, model_name="ZeroGumerNet",
                              train_ds=train_ds, class_weights=class_weights,
                              val_ds=val_ds,
                              dir_tosave=second_strategy_dir,
                              epochs=epochs, lr=lr,
                              es_patience=es_patience, lr_patience=lr_patience,
                              lr_factor=lr_factor, min_lr=min_lr)

    if make_plots:
        history_csv_dir = os.path.join(second_strategy_dir, "ZeroGumerNet_training_history.csv")
        plot_loss_and_accuracy(base_dir=second_strategy_dir, model_name="ZeroGumerNet_best.keras", history_csv_dir=history_csv_dir)


def compute_metrics(model_path: str, dataset, class_names: list):
    """
        Given a model makes predictions and compute F1 score.
        Returns also the classification report, y_true and y_pred.
    """

    model = load_model(model_path)

    preds = model.predict(dataset, verbose=0)
    y_pred = np.argmax(preds, axis=-1)

    labels_list = [labels for _, labels in dataset]
    y_true = np.argmax(np.concatenate(labels_list, axis=0), axis=-1)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    macro_f1 = report_dict["macro avg"]["f1-score"]

    return macro_f1, report_dict, y_true, y_pred


def model_selection():
    """
        Takes the models of the two strategies and compute the F1 score on the validation set.
        Then select the best.

        The best model is used to make predictions on the test set.
        At the end is saved the classification report and the confusion matrix computed on the test set.
    """

    finetuning_dir = os.path.join(os.getcwd(), "checkpoints", "first_strategy", "finetuning")
    second_strategy_dir = os.path.join(os.getcwd(), "checkpoints", "second_strategy")

    # Load VAL set
    val_ds_path = os.path.join(second_strategy_dir, "val_ds")
    val_ds = tf.data.Dataset.load(val_ds_path)

    first_strategy_model = os.path.join(finetuning_dir, "GumerNet_best.keras")
    second_strategy_model = os.path.join(second_strategy_dir, "ZeroGumerNet_best.keras")

    # Load TEST set
    stratified_dataset_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse_stratified")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(stratified_dataset_dir, "test"),
        seed=177,
        shuffle=False,
        image_size=(224, 224),
        batch_size=16,
        label_mode='categorical'
    )
    class_names = test_ds.class_names

    # Calculate metrics on VAL set
    f1_gumer, report_dict, y_true, y_pred = compute_metrics(first_strategy_model, val_ds, class_names)
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(finetuning_dir, "classification_report_on_val_UC.csv")
    df_report.to_csv(csv_path)

    f1_zero, report_dict, y_true, y_pred = compute_metrics(second_strategy_model, val_ds, class_names)
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(second_strategy_dir, "classification_report_on_val_UC.csv")
    df_report.to_csv(csv_path)

    # Select best model
    if f1_gumer > f1_zero:
        best_model_path = first_strategy_model
        best_model_name = "GumerNet"
    else:
        best_model_path = second_strategy_model
        best_model_name = "ZeroGumerNet"

    # Save best model a part
    best_model_dir = os.path.join(os.getcwd(), "checkpoints", "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    shutil.copy(best_model_path, os.path.join(best_model_dir, f"{best_model_name}_best.keras"))
    shutil.copy(best_model_path, os.path.join(os.getcwd(), "best_model.keras"))

    # Save class_names a part
    class_names_path = os.path.join(second_strategy_dir, "class_names.pkl")
    shutil.copy(class_names_path, os.path.join(os.getcwd(), "class_names.pkl"))

    _, report_dict, y_true, y_pred = compute_metrics(best_model_path, test_ds, class_names)

    # Save the classification report for the best model
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(best_model_dir, "classification_report.csv")
    df_report.to_csv(csv_path)

    # Confusion matrix for the TEST set
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation='vertical', cmap="hot",
                                                                values_format='.0f')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig(os.path.join(best_model_dir, f"confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save report on TEST for both strategy
    f1_gumer, report_dict, y_true, y_pred = compute_metrics(first_strategy_model, test_ds, class_names)
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(finetuning_dir, "classification_report_on_test_UC.csv")
    df_report.to_csv(csv_path)

    f1_zero, report_dict, y_true, y_pred = compute_metrics(second_strategy_model, test_ds, class_names)
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(second_strategy_dir, "classification_report_on_test_UC.csv")
    df_report.to_csv(csv_path)


def single_img_classification(image_path: str):
    """
        Use best model to classify image
    """

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_batch = np.expand_dims(img_resized, axis=0)

    model_path = os.path.join(os.getcwd(), "best_model.keras")
    model = load_model(model_path)
    preds = model.predict(img_batch, verbose=0)

    class_names_file = os.path.join(os.getcwd(), "class_names.pkl")
    class_names = pickle.load(open(class_names_file, "rb"))

    predicted_class_index = np.argmax(preds[0])
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name, preds[0], class_names


def plot_probs(probs: np.ndarray, class_names: list):
    """
        Plot the 5 most probable classes with their probs.
    """

    top_5_idx = np.argsort(probs)[-5:][::-1]
    top_5_probs = probs[top_5_idx]
    top_5_classes = [class_names[i] for i in top_5_idx]

    top_5_probs = top_5_probs[::-1]
    top_5_classes = top_5_classes[::-1]

    colors = ['#FEE2E2', '#FCA5A5', '#F87171', '#DC2626',  '#7F1D1D']
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(top_5_classes, top_5_probs, color=colors, height=0.6)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{width * 100:.1f}%',
                va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')

    ax.grid(False)
    ax.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig("probs.png", dpi=300, bbox_inches='tight')
    plt.close()




if __name__ == "__main__":

    # Initialize GPU
    setup_gpu()


    # FIRST STRATEGY #
    pre_training_phase = False
    # PRETRAINING PHASE
    if pre_training_phase:
        pre_training(train=True, make_plots=True)

    finetuning_phase = False
    # FINETUNING PHASE
    if finetuning_phase:
        finetuning_step(train=True, make_plots=True)


    # SECOND STRATEGY #
    second_strategy_phase = False
    if second_strategy_phase:
        second_strategy(train=True, make_plots=True)


    # MODEL SELECTION #
    model_selection_phase = False
    if model_selection_phase:
        model_selection()


    # INFERENCE
    img_path = "punta_raisi.png"
    label, all_probs, class_names = single_img_classification(img_path)
    print(f"\n<- Classifico {os.path.basename(img_path)} ->")
    print(f"<- L'immagine fornita appartiene alla classe: {label} ->")
    plot_probs(all_probs, class_names)
