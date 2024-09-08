import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument(
        "--image_height", type=int, default=576, help="Height of input images"
    )
    parser.add_argument(
        "--image_width", type=int, default=672, help="Width of input images"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of channels in input images"
    )
    parser.add_argument(
        "--image_mean",
        type=float,
        default=128.12657067878447,
        help="Mean value for image normalization",
    )
    parser.add_argument(
        "--total_classes", type=int, default=7, help="Total number of classes"
    )
    parser.add_argument(
        "--end_activation",
        type=str,
        default="softmax",
        help="Activation function for the output layer",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs to train"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=17283,
        help="Number of samples in the training dataset",
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=5289,
        help="Number of samples in the validation dataset",
    )
    parser.add_argument(
        "--train_tfrecord",
        type=str,
        default="D:/LS3_LPC/Data/Train/train.tfrecords",
        help="Path to training TFRecord",
    )
    parser.add_argument(
        "--val_tfrecord",
        type=str,
        default="D:/LS3_LPC/Data/Val/val.tfrecords",
        help="Path to validation TFRecord",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="D:/LS3_LPC/Checkpoints/Checkpoint-{epoch:04d}.hdf5",
        help="Path to save model checkpoints",
    )
    parser.add_argument(
        "--train_log_path",
        type=str,
        default="D:/LS3_LPC/train_logs.csv",
        help="Path to save training logs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    """General inputs and hyperparameters"""
    image_height = args.image_height
    image_width = args.image_width
    channels = args.channels

    image_mean = args.image_mean
    total_classes = args.total_classes

    end_activation = args.end_activation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs

    train_samples = args.train_samples
    val_samples = args.val_samples

    train_tfrecord = args.train_tfrecord
    val_tfrecord = args.val_tfrecord

    checkpoint_path = args.checkpoint_path
    train_log_path = args.train_log_path

    # Print out the values to confirm they are accessible inside the container
    print(f"Image Height: {image_height}")
    print(f"Image Width: {image_width}")
    print(f"Channels: {channels}")
    print(f"Image Mean: {image_mean}")
    print(f"Total Classes: {total_classes}")
    print(f"End Activation: {end_activation}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Train Samples: {train_samples}")
    print(f"Validation Samples: {val_samples}")
    print(f"Train TFRecord Path: {train_tfrecord}")
    print(f"Validation TFRecord Path: {val_tfrecord}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Train Log Path: {train_log_path}")


if __name__ == "__main__":
    main()
