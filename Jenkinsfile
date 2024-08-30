pipeline{
    agent any 

    parameters {
        string(name: 'IMAGE_HEIGHT', defaultValue: '576', description: 'Height of input images')
        string(name: 'IMAGE_WIDTH', defaultValue: '672', description: 'Width of input images')
        string(name: 'CHANNELS', defaultValue: '1', description: 'Number of channels in input images')

        string(name: 'IMAGE_MEAN', defaultValue: '128.12657067878447', description: 'Mean value for image normalization')
        string(name: 'TOTAL_CLASSES', defaultValue: '7', description: 'Total number of classes')

        choice(name: 'END_ACTIVATION', choices: ['softmax', 'sigmoid', 'linear'], description: 'Activation function for the output layer')
        
        string(name: 'LEARNING_RATE', defaultValue: '0.0001', description: 'Learning rate')
        string(name: 'BATCH_SIZE', defaultValue: '8', description: 'Batch size')
        string(name: 'EPOCHS', defaultValue: '300', description: 'Number of epochs to train')

        string(name: 'TRAIN_SAMPLES', defaultValue: '17283', description: 'Number of samples in the training dataset')
        string(name: 'VAL_SAMPLES', defaultValue: '5289', description: 'Number of samples in the validation dataset')

        string(name: 'TRAIN_TFRECORD', defaultValue: 'D:/LS3_LPC/Data/Train/train.tfrecords', description: 'Path to training TFRecord')
        string(name: 'VAL_TFRECORD', defaultValue: 'D:/LS3_LPC/Data/Val/val.tfrecords', description: 'Path to validation TFRecord')
        string(name: 'CHECKPOINT_PATH', defaultValue: 'D:/LS3_LPC/Checkpoints/Checkpoint-{epoch:04d}.hdf5', description: 'Path to save model checkpoints')
        string(name: 'TRAIN_LOG_PATH', defaultValue: 'D:/LS3_LPC/train_logs.csv', description: 'Path to save training logs')
    }

    stages{

        stage('Parameters Extraction and Trimming'){
            steps{
                script{
                    // Extract and Trim the parameters 
                    def IMAGE_HEIGHT = params.IMAGE_HEIGHT.trim()
                    def IMAGE_WIDTH = params.IMAGE_WIDTH.trim()
                    def CHANNELS = params.CHANNELS.trim()

                    def IMAGE_MEAN = params.IMAGE_MEAN.trim()            
                    def TOTAL_CLASSES = params.TOTAL_CLASSES.trim()

                    // Set default value after defining the parameter
                    // params.END_ACTIVATION = params.END_ACTIVATION ?: 'softmax'
                    def END_ACTIVATION = params.END_ACTIVATION.trim()

                    def LEARNING_RATE = params.LEARNING_RATE.trim()
                    def BATCH_SIZE = params.BATCH_SIZE.trim()
                    def EPOCHS = params.EPOCHS.trim()

                    def TRAIN_SAMPLES = params.TRAIN_SAMPLES.trim() 
                    def VAL_SAMPLES = params.VAL_SAMPLES.trim()                     

                    def TRAIN_TFRECORD = params.TRAIN_TFRECORD.trim()
                    def VAL_TFRECORD = params.VAL_TFRECORD.trim()
                    def CHECKPOINT_PATH = params.CHECKPOINT_PATH.trim()
                    def TRAIN_LOG_PATH = params.TRAIN_LOG_PATH.trim()

                    echo "IMAGE_HEIGHT : ${IMAGE_HEIGHT}"
                    echo "IMAGE_WIDTH : ${IMAGE_WIDTH}"
                    echo "CHANNELS : ${CHANNELS}"

                    echo "IMAGE_MEAN : ${IMAGE_MEAN}"
                    echo "TOTAL_CLASSES : ${TOTAL_CLASSES}"

                    echo "END_ACTIVATION : ${END_ACTIVATION}"
                    
                    echo "LEARNING_RATE : ${LEARNING_RATE}"
                    echo "BATCH_SIZE : ${BATCH_SIZE}"
                    echo "EPOCHS : ${EPOCHS}"

                    echo "TRAIN_SAMPLES  : ${TRAIN_SAMPLES}"                    
                    echo "VAL_SAMPLES  : ${VAL_SAMPLES}"
                    
                    echo "TRAIN_TFRECORD : ${TRAIN_TFRECORD}"
                    echo "VAL_TFRECORD : ${VAL_TFRECORD}"
                    echo "CHECKPOINT_PATH : ${CHECKPOINT_PATH}" 
                    echo "TRAIN_LOG_PATH : ${TRAIN_LOG_PATH}" 
                }
            }
        }
    }
        stage('Build Docker Image') {
            steps {
                script {
                    sh 'docker build -t my_ml_training_image .'
                }
            }
        }

        stage('Run TensorFlow Training') {
            agent {
                docker {
                    image 'my_ml_training_image'
                    args '-u root --gpus all'
                    }
                }
                steps {
                    script {
                        sh """
                        python test.py --train_samples=${TRAIN_SAMPLES} \
                                        --val_samples=${VAL_SAMPLES} \
                                        --epochs=${EPOCHS} \
                                        --batch_size=${BATCH_SIZE} \
                                        --learning_rate=${LEARNING_RATE} \
                                        --train_tfrecord=${TRAIN_TFRECORD} \
                                        --val_tfrecord=${VAL_TFRECORD} \
                                        --checkpoint_path=${CHECKPOINT_PATH} \
                                        --train_log_path=${TRAIN_LOG_PATH} \
                                        --image_mean=${IMAGE_MEAN} \
                                        --total_classes=${TOTAL_CLASSES} \
                                        --end_activation=${END_ACTIVATION}
                        """
                        }
                    }
                }
    post {
        always {
            echo 'Pipeline completed!'
        }
        success {
            echo 'Pipeline succeeded!'
            // Send success notification
        }
        failure {
            echo 'Pipeline failed!'
            // Send failure notification
        }
    }


}