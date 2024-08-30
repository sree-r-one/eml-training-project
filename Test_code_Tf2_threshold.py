import tensorflow as tf
import numpy as np
import time
import os
import shutil
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    if print_graph:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

def resize_image(image_path, labels):
    '''Perform preprocessing of image using tf. Resize image using tensorflow '''
    raw_image = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_bmp(raw_image)
    image_resized = tf.image.resize(image_tensor, size, method='area')
    image_resized = (image_resized - img_mean) / img_std_dev
    return image_resized, labels, image_path

def get_key(val):
    '''Function to obtain Key based on index value, as defined in dict '''
    for key, value in class_label_dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def extract_images(root_dir, move_img=True, logs=True):
    '''obtain predictions for image'''
    image_files = [os.path.join(dir, name) for dir, _, files in os.walk(root_dir) for name in files]
    list_classes = [os.path.split(os.path.split(i)[0])[1] for i in image_files]
    image_labels = [class_label_dict.get(i) for i in list_classes]
    
    # Counting images per class
    image_count_per_class = {class_name: list_classes.count(class_name) for class_name in set(list_classes)}
    print("Image count per class:", image_count_per_class)
    
    total_num_images = len(image_files)
    print(f"Total number of images present : {total_num_images}")
    
    dataset = tf.data.Dataset.from_tensor_slices((image_files, image_labels))
    dataset = dataset.map(resize_image, num_parallel_calls=4)
    dataset = dataset.batch(2)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    confusion_matrix = np.zeros(shape=(num_of_classes, num_of_classes))
    img_cnt = 0
    cnt = 0
    while cnt < total_num_images:
        image, lbls, paths = iterator.get_next()
        for i in range(image.shape[0]):
            path = paths[i].numpy().decode("utf-8") # convert tensor format to array and bytes string to string
            img = tf.expand_dims(image[i], axis=0)

            # Get predictions
            true_label = lbls[i].numpy()
            frozen_graph_predictions = frozen_func(x=tf.constant(img))[0]
            predictions = frozen_graph_predictions[0].numpy().astype('float32')
            
            threshold_prediction = [predictions[i] if predictions[i] >= thresholds[i] else 0 for i in range(len(predictions))]

            for i in range(len(predictions)):
                for i in range(len(thresholds)):
                    if predictions[i] > thresholds [i]:
                        predictions[i]
                    else:
                        0

            predicted_label = []
            if np.count_nonzero(threshold_prediction) == 0:
                predicted_label.append(np.argmax(predictions))                
            
            
            else:
                indices_above_threshold = [i for i, value in enumerate(threshold_prediction) if value > 0]
                if indices_above_threshold:
                    highest_prediction_index = max(indices_above_threshold, key=lambda index: predictions[index])
                    predicted_label.append(highest_prediction_index)
                    move_image_based_on_prediction(path, highest_prediction_index)
                    
            img_cnt += 1
            confusion_matrix[true_label][predicted_label[0]] += 1
            #
            # if logs:
            #     with open(img_predictions, 'a') as fo:
            #         fo.write(f"{img_cnt}, {path.split('\\')[-1]}, {predictions}\n")
            #     with open(threshold_predictions, 'a') as fo:
            #         fo.write(f"{img_cnt}, {path.split('\\')[-1]}, {threshold_prediction}\n")

            if move_img:
                move_image_based_on_prediction(path, predicted_label[0])

        cnt += image.shape[0]

    print_confusion_matrix_and_stats(confusion_matrix, image_count_per_class)
    return confusion_matrix

def move_image_based_on_prediction(path, predicted_label):
    true_class_name = os.path.split(path.split('\\')[-2])[-1]
    predicted_class_name = get_key(predicted_label)
    dst_true_class = os.path.join(move_dir, true_class_name)
    dst_predicted_class = os.path.join(dst_true_class, predicted_class_name)

    if not os.path.isdir(move_dir):
        os.mkdir(move_dir)
    if not os.path.isdir(dst_true_class):
        os.mkdir(dst_true_class)
    if not os.path.isdir(dst_predicted_class):
        os.mkdir(dst_predicted_class)

    shutil.copy(path, os.path.join(dst_predicted_class, path.split('\\')[-1]))

def print_confusion_matrix_and_stats(confusion_matrix, image_count_per_class):
    acc = []
    prec = []
    for i, class_name in enumerate(list_class):
        acc_ = confusion_matrix[i][i] / image_count_per_class[class_name] if class_name in image_count_per_class else 0
        prec_ = confusion_matrix[i][i] / sum(confusion_matrix[:, i]) if sum(confusion_matrix[:, i]) > 0 else 0
        acc.append(acc_)
        prec.append(prec_)
        if acc_ < 1:
            for j, num in enumerate(confusion_matrix[i]):
                if num != confusion_matrix[i][i] and num != 0:
                    print(f"Misclassified as {list_class[j]} with count {confusion_matrix[i][j]}")

    df_conf_mat = pd.DataFrame(confusion_matrix, index=[i for i in list_class], columns=[c for c in list_class])
    df_conf_mat.loc['Precision'] = prec
    df_conf_mat = pd.concat([df_conf_mat, pd.Series(acc, index=[i for i in list_class], name='Accuracy')], axis=1)
    with open(confus_csv, 'a') as f:
        df_conf_mat.to_csv(f)

if __name__ == "__main__":
    # Class labels dictionary

    class_label_dict = {'Lens out of POV':0,
                        'Low Saline':1,
                        'Multiple Lens':2,
                        'No Lens':3,
                        'No Shell':4,
                        'Normal Lens':5,
                        'Shell out of POV':6}

    # List of class names
    list_class = list(class_label_dict.keys())

    # Thresholds for each class, for prediction filtering
    thresholds = [0.2] * len(list_class)

    # Directory paths
    root_dir = "D:/chinna/LS3_LPC/Data/Val/"                                    #"D:/subbu/FCI/Data/Val/"
    graph_path = "D:/chinna/LS3_LPC/Checkpoints/results/Checkpoint-0141.pb"     #"D:/subbu/FCI/Data/results/Checkpoint-0221.pb"

    # Control flags
    move_imgs = True
    logs = True

    # Output log files paths
    #img_predictions = "D:/subbu/FCI/Data/results/221/pred_221.txt"
    #threshold_predictions =  "D:/subbu/FCI/Data/results/221/Thers_221.txt"
    confus_csv = "D:/chinna/LS3_LPC/Checkpoints/results/141/conf_141.csv"   #'D:/subbu/FCI/Data/results/221/Conf_221.csv'
    move_dir = "D:/chinna/LS3_LPC/Checkpoints/results/141"                  #"D:/subbu/FCI/Data/results/221/"

    # Image preprocessing parameters
    img_mean = 122.24150242883458                         #121.53421898775322
    img_std_dev = 255
    size = (576, 672)

    # Tensor names in the frozen graph
    input_name = "x:0"
    output_name = "Identity:0"

    # Number of classes
    num_of_classes = len(class_label_dict)

    # Load the TensorFlow graph
    print("Loading Graph...")
    start = time.time()
    with tf.io.gfile.GFile(graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())
    end = time.time()
    print(f"Finished Loading graph in {end - start} seconds")

    # Convert graph def to a concrete function
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=[input_name],
                                    outputs=[output_name],
                                    print_graph=False)

    # Process images and get confusion matrix
    confusion_matrix = extract_images(root_dir, move_imgs, logs)
    end = time.time()
    print(f"Finished Testing : in {end - start} seconds")

