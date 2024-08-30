
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import time 

print("tensorflow version currently in use:",tf.__version__)
print("GPU name " ,tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


def convert_pb(model_path, save_path, save_logs, print_graph=False):
    model = tf.keras.models.load_model(model_path)
    print(f"Model input : {model.inputs}") # [<tf.Tensor 'input_1:0' shape=(None, 512, 611, 1) dtype=float32>]
    print(f"Model output : {model.outputs}") # [<tf.Tensor 'dense/Softmax:0' shape=(None, 9) dtype=float32>]

    # Convert Keras model to ConcreteFunction ,since usage of tf.graph directly is deprecated in tf2.0
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
                 x = tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # print("get concerete func :",full_model)

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    if print_graph == True:
        print("-" * 50)
        print(f"Frozen model layers: \n{layers} ")
        for layer in layers:
            print(layer)

    print("-" * 50)
    print(f"Frozen model inputs : {frozen_func.inputs}")  #[<tf.Tensor 'x:0' shape=(None, 512, 611, 1) dtype=float32>]
    print(f"Frozen model outputs : {frozen_func.outputs}") #[<tf.Tensor 'Identity:0' shape=(None, 9) dtype=float32>]

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def = frozen_func.graph,
                      logdir = save_logs,
                      name = save_path,
                      as_text = False)

if __name__ == "__main__":
    # filename = LS3_bv_512_052
    model_path = "D:/chinna/LS3_LPC/Checkpoints/Checkpoint-0141.hdf5"            #"D:/FCI/Data/Checkpoints/Checkpoint-0231.hdf5"                   ##"D:/Swati/FCI/Checkpoints/ckpt34/Checkpoint-0074.hdf5"
    save_path = "D:/chinna/LS3_LPC/Checkpoints/results/Checkpoint-0141.pb"       #"D:/FCI/Data/Checkpoints/results/Checkpoint-0231.pb"              #"D:/Swati/FCI/Checkpoints/ckpt34/results/Checkpoint-0073.pb"
    save_logs = "D:/chinna/LS3_LPC/Checkpoints/"                                 #"D:/FCI/Data/Checkpoints/"                                        #"D:/Swati/FCI/Checkpoints/"
    
    print(f"starting Conversion from h5 to pb :")

    start = time.time() 
    pb = convert_pb(model_path, save_path, save_logs)
    end = time.time() 
    print(f"Completed file conversion in {end-start} seconds")