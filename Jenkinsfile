pipeline {
    agent {
        label 'eml-project'
    }

    parameters {
        // Image parameters
        string(name: 'IMAGE_HEIGHT', defaultValue: '576', description: 'Height of input images')
        string(name: 'IMAGE_WIDTH', defaultValue: '672', description: 'Width of input images')
        string(name: 'CHANNELS', defaultValue: '1', description: 'Number of channels in input images')

        // Data normalization and model parameters
        string(name: 'IMAGE_MEAN', defaultValue: '128.12657067878447', description: 'Mean value for image normalization')
        string(name: 'TOTAL_CLASSES', defaultValue: '7', description: 'Total number of classes')
        choice(name: 'END_ACTIVATION', choices: ['softmax', 'sigmoid', 'linear'], description: 'Activation function for the output layer')

        // Training hyperparameters
        string(name: 'LEARNING_RATE', defaultValue: '0.0001', description: 'Learning rate')
        string(name: 'BATCH_SIZE', defaultValue: '8', description: 'Batch size')
        string(name: 'EPOCHS', defaultValue: '300', description: 'Number of epochs to train')

        // Dataset parameters
        string(name: 'TRAIN_SAMPLES', defaultValue: '17283', description: 'Number of samples in the training dataset')
        string(name: 'VAL_SAMPLES', defaultValue: '5289', description: 'Number of samples in the validation dataset')

        // File paths
        string(name: 'TRAIN_TFRECORD', defaultValue: 'D:/LS3_LPC/Data/Train/train.tfrecords', description: 'Path to training TFRecord')
        string(name: 'VAL_TFRECORD', defaultValue: 'D:/LS3_LPC/Data/Val/val.tfrecords', description: 'Path to validation TFRecord')
        string(name: 'CHECKPOINT_PATH', defaultValue: 'D:/LS3_LPC/Checkpoints/Checkpoint-{epoch:04d}.hdf5', description: 'Path to save model checkpoints')
        string(name: 'TRAIN_LOG_PATH', defaultValue: 'D:/LS3_LPC/train_logs.csv', description: 'Path to save training logs')
    }

    stages {

        stage('Parameters Extraction and Trimming') {
            steps {
                script {
                    // Extract and Trim the parameters 
                    env.IMAGE_HEIGHT = params.IMAGE_HEIGHT.trim()
                    env.IMAGE_WIDTH = params.IMAGE_WIDTH.trim()
                    env.CHANNELS = params.CHANNELS.trim()

                    env.IMAGE_MEAN = params.IMAGE_MEAN.trim()
                    env.TOTAL_CLASSES = params.TOTAL_CLASSES.trim()

                    env.END_ACTIVATION = params.END_ACTIVATION.trim()

                    env.LEARNING_RATE = params.LEARNING_RATE.trim()
                    env.BATCH_SIZE = params.BATCH_SIZE.trim()
                    env.EPOCHS = params.EPOCHS.trim()

                    env.TRAIN_SAMPLES = params.TRAIN_SAMPLES.trim() 
                    env.VAL_SAMPLES = params.VAL_SAMPLES.trim()

                    env.TRAIN_TFRECORD = params.TRAIN_TFRECORD.trim()
                    env.VAL_TFRECORD = params.VAL_TFRECORD.trim()
                    env.CHECKPOINT_PATH = params.CHECKPOINT_PATH.trim()
                    env.TRAIN_LOG_PATH = params.TRAIN_LOG_PATH.trim()

                    // Echo the parameter values for verification
                    echo "IMAGE_HEIGHT : ${env.IMAGE_HEIGHT}"
                    echo "IMAGE_WIDTH : ${env.IMAGE_WIDTH}"
                    echo "CHANNELS : ${env.CHANNELS}"

                    echo "IMAGE_MEAN : ${env.IMAGE_MEAN}"
                    echo "TOTAL_CLASSES : ${env.TOTAL_CLASSES}"

                    echo "END_ACTIVATION : ${env.END_ACTIVATION}"
                    
                    echo "LEARNING_RATE : ${env.LEARNING_RATE}"
                    echo "BATCH_SIZE : ${env.BATCH_SIZE}"
                    echo "EPOCHS : ${env.EPOCHS}"

                    echo "TRAIN_SAMPLES : ${env.TRAIN_SAMPLES}"
                    echo "VAL_SAMPLES : ${env.VAL_SAMPLES}"
                    
                    echo "TRAIN_TFRECORD : ${env.TRAIN_TFRECORD}"
                    echo "VAL_TFRECORD : ${env.VAL_TFRECORD}"
                    echo "CHECKPOINT_PATH : ${env.CHECKPOINT_PATH}" 
                    echo "TRAIN_LOG_PATH : ${env.TRAIN_LOG_PATH}" 
                }
            }
        }
        stage('Run Training') {
            steps {
                // script {
                //     // Command to run train.py with Jenkins parameters as command-line arguments
                //     sh """
                //         python3 test.py \
                //         --image_height ${env.IMAGE_HEIGHT} \
                //         --image_width ${env.IMAGE_WIDTH} \
                //         --channels ${env.CHANNELS} \
                //         --image_mean ${env.IMAGE_MEAN} \
                //         --total_classes ${env.TOTAL_CLASSES} \
                //         --end_activation ${env.END_ACTIVATION} \
                //         --learning_rate ${env.LEARNING_RATE} \
                //         --batch_size ${env.BATCH_SIZE} \
                //         --epochs ${env.EPOCHS} \
                //         --train_samples ${env.TRAIN_SAMPLES} \
                //         --val_samples ${env.VAL_SAMPLES} \
                //         --train_tfrecord ${env.TRAIN_TFRECORD} \
                //         --val_tfrecord ${env.VAL_TFRECORD} \
                //         --checkpoint_path ${env.CHECKPOINT_PATH} \
                //         --train_log_path ${env.TRAIN_LOG_PATH}
                //     """
                // }
            }
        }

    }

    post {
        always {
            echo 'Pipeline completed!' 
        }
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
