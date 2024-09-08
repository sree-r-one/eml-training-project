import tensorflow as tf
from SE_inception_v3 import SE_InceptionV3
from read_tfrecord import Read_TFRecord

#############
import os

os.environ["CUDA_VISILE_DEVICES"] = "0"
#############


def main():
    """General inputs and hyperparameters"""
    ##################################################################################################################

    image_height = 576
    image_width = 672
    channels = 1

    image_mean = 128.12657067878447
    total_classes = 7

    end_activation = "softmax"

    learning_rate = 0.0001
    batch_size = 8
    epochs = 300

    train_samples = 17283
    val_samples = 5289

    train_tfrecord = "D:/LS3_LPC/Data/Train/train.tfrecords"
    val_tfrecord = "D:/LS3_LPC/Data/Val/val.tfrecords"

    checkpoint_path = "D:/LS3_LPC/Checkpoints/Checkpoint-{epoch:04d}.hdf5"
    train_log_path = "D:/LS3_LPC/train_logs.csv"
    ##################################################################################################################
    # os.makedirs(train_log_path, exist_ok=True)

    initial_epoch = 0  # this if you are resuming training and put the number from what u want start training
    # here kept 1 because we loaded epoch 1, we want to save from epoch number 2

    """ If You are uncommenting this line then comment out the Line number 63,64,65 compile inception model and summary lines"""

    # model = tf.keras.models.load_model('D:/LPC_classification/Harsha/SE INV3/ckpts/Checkpoint-0060.hdf5')  # provide the Path of h5 file
    # model = tf.keras.models.load_model('D:/chinna/LS3_LPC/Checkpoints/Checkpoint-0144.hdf5')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_freq="epoch", period=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger(train_log_path)

    extract_images = Read_TFRecord(
        image_size=(image_height, image_width, channels), image_mean=image_mean
    )

    print("\nExtracting Train Images")
    train_data = extract_images.decode_tfrecord(
        train_tfrecord, batch_size, mode="train"
    )
    print("\nExtracting Validation Images")
    val_data = extract_images.decode_tfrecord(val_tfrecord, batch_size, mode="val")

    train_steps = train_samples // batch_size
    val_steps = val_samples // batch_size

    model = SE_InceptionV3(
        input_shape=(image_height, image_width, channels),
        classes=total_classes,
        classifier_activation=end_activation,
    )

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=val_data,
        validation_steps=val_steps,
        verbose=1,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_callback, csv_logger],
    )


if __name__ == "__main__":
    main()
