
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import time

class Read_TFRecord:
    def __init__(self, image_size, image_mean):
        self.IMAGE_SIZE = image_size
        self.image_mean = image_mean

    def extract_fn(self, data_record):
        feature_set = {'label': tf.io.FixedLenFeature([], tf.int64),
                       'image': tf.io.FixedLenFeature([], tf.string)}

        sample = tf.io.parse_single_example(data_record, feature_set)

        """While creating tfrecord, 
        if images are in uint8 then need to use tf.uint8 in decode_raw, and then type cast it to tf.float32
        if images are in float32, then need to use tf.float32 in decode_raw."""
        # image = tf.decode_raw(sample['image'], tf.float32)
        # image = tf.reshape(image, self.IMAGE_SIZE)
        image = tf.io.decode_raw(sample['image'], tf.uint8)
        image = tf.dtypes.cast(image, tf.float32)
        image = tf.reshape(image, self.IMAGE_SIZE)
        image = (image - self.image_mean)/255
        label = tf.cast(sample['label'], tf.int64)

        return  image, label
    @tf.autograph.experimental.do_not_convert
    def decode_tfrecord(self, tfrecord_loc, batch_size, mode):
        """Decodes tfrecord in path specified by data_path"""

        # reading tfrecord file and extracting images and labels
        file_name = tf.data.Dataset.list_files(tfrecord_loc)

        if mode is "train" and mode is not "val":
            tfrecord_dataset = file_name \
                .shuffle(buffer_size=5000).repeat()\
                .interleave(lambda x: tf.data.TFRecordDataset(x).map(self.extract_fn), cycle_length=2, block_length=4,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(batch_size)

        if mode is "val" and mode is not "train":
            tfrecord_dataset = file_name \
                .interleave(lambda x: tf.data.TFRecordDataset(x).map(self.extract_fn), cycle_length=2, block_length=4) \
                .repeat().batch(batch_size)

        tfrecord_dataset = tfrecord_dataset.prefetch(1)

        return tfrecord_dataset

