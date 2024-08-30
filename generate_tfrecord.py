import os
import tensorflow as tf
import time
import numpy as np
from tqdm import tqdm


def resize_image(image_path):
    print("image_path",image_path)
    raw_image = tf.io.read_file(image_path)  
    image_tensor = tf.image.decode_bmp(raw_image)
    img_4d = tf.expand_dims(image_tensor, axis=0)  # coz resize_bicubic needs 4D tensor
    #img_inter_area = tf.image.resize_bicubic(img_4d, (536, 640))
    img_inter_area = tf.image.resize(img_4d, (576,672))
    image_resized = tf.squeeze(img_inter_area, [0])
    image_resized =tf.cast(image_resized, tf.uint8)

    return image_resized, image_path


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_creation(label, image):
    feature = {'label': _int64_feature(label),
               'image': _bytes_feature(tf.compat.as_bytes(image.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


def create_tfrecord(root_dir, tfrecord_loc, batch_count):
    available_class = [e for e in os.listdir(root_dir)]

    class_label_dict = {'Lens out of POV':0,
                        'Low Saline':1,
                        'Multiple Lens':2,
                        'No Lens':3,
                        'No Shell':4,
                        'Normal Lens':5,
                        'Shell out of POV':6}

    list_class = [e for e in available_class if e in class_label_dict]

    for i, class_name in enumerate(list_class):
        print("Label corresponding to class \t{:<20}\t is {:<3}".format(class_name, class_label_dict.get(class_name)))

    print("{} CLASSES ARE NOT OF OUR INTEREST".format([e for e in available_class if e not in list_class]))

    interested_class_path = [root_dir + '\\' + e + '\\*.bmp' for e in list_class]

    mean_array = []

    all_images_path = tf.io.gfile.glob(interested_class_path)

    total_no_of_samples = len(all_images_path)

    print("We have {} interested images".format(total_no_of_samples))

    # give_class = lambda x: os.path.split(os.path.split(x)[0])[-1].decode("utf-8")
    give_class = lambda x: os.path.split(os.path.split(x)[0])[-1].decode("utf-8")
    writer = tf.io.TFRecordWriter(tfrecord_loc)

    files = tf.data.Dataset.from_tensor_slices(all_images_path).shuffle(total_no_of_samples).batch(1024)
    files = files.interleave(lambda x: tf.data.Dataset.list_files(x).map(resize_image, num_parallel_calls=8),
                             cycle_length=32,
                             block_length = 4)#.batch(batch_count)
    files = files.prefetch(buffer_size=1)
    # it = files.make_one_shot_iterator()

    # next_file = it.get_next()

    count = 0
    start_time = time.time()

    print("Started creating tfrecord....")

    # with tf.Session() as sess:

    try:
        # while count < total_no_of_samples:
        # Replace the existing loop with this updated loop
        for ii in tqdm(files.as_numpy_iterator(), total=total_no_of_samples):
            image, path = ii

            writer.write(feature_creation(class_label_dict.get(give_class(path)), image))
            mean_array.append(np.mean(image))
            count = count + 1

            if not count % (4 * batch_count):
                print("Finished writing {:<6}th image with class {} time {}".format(count, give_class(path),
                                                                                    (time.time() - start_time) / 60))
                start_time = time.time()


    except:
        raise ValueError("Exiting Process - End of Sequence")
        pass

    print("Wrote {} images".format(count))
    return np.mean(mean_array)


if __name__ == "__main__":
    root_dir = "D:/LS3_LPC/Data/Train"
    tfrecord_loc = 'E:/LS3_LPC/Data/Train/train.tfrecords'

    start_time_ = time.time()
    mean = create_tfrecord(root_dir, tfrecord_loc, batch_count=32)
    print(f"Mean of Images {mean}")
    print("Took {} min to create tfrecord".format((time.time() - start_time_) / 60))
