pipeline{
    agent any 

    parameters {
        string(name: 'EPOCHS', defaultValue: '300', description: 'Number of epochs to train')
        string(name: 'BATCH_SIZE', defaultValue: '8', description: 'Batch size')
        string(name: 'LEARNING_RATE', defaultValue: '0.0001', description: 'Learning rate')
        string(name: 'TRAIN_TFRECORD', defaultValue: 'D:/LS3_LPC/Data/Train/train.tfrecords', description: 'Path to training TFRecord')
        string(name: 'VAL_TFRECORD', defaultValue: 'D:/LS3_LPC/Data/Val/val.tfrecords', description: 'Path to validation TFRecord')
        string(name: 'IMAGE_HEIGHT', defaultValue: '576', description: 'Height of input images')
        string(name: 'IMAGE_WIDTH', defaultValue: '672', description: 'Width of input images')
        string(name: 'CHANNELS', defaultValue: '1', description: 'Number of channels in input images')
    }

    stages{
        stage('Parameters Extraction and Trimming'){
            steps{
                script{
                    // Extract and Trim the parameters 
                    def EPOCHS = params.EPOCHS.trim()
                    def BATCH_SIZE = params.BATCH_SIZE.trim()
                    def LEARNING_RATE = params.LEARNING_RATE.trim()
                    def TRAIN_TFRECORD = params.TRAIN_TFRECORD.trim()
                    def VAL_TFRECORD = params.VAL_TFRECORD.trim()
                    def IMAGE_HEIGHT = params.IMAGE_HEIGHT.trim()
                    def IMAGE_WIDTH = params.IMAGE_WIDTH.trim()
                    def CHANNELS = params.CHANNELS.trim()

                    echo "EPOCHS : ${EPOCHS}"
                    echo "BATCH_SIZE : ${BATCH_SIZE}"
                    echo "LEARNING_RATE : ${LEARNING_RATE}"
                    echo "TRAIN_TFRECORD : ${TRAIN_TFRECORD}"
                    echo "VAL_TFRECORD : ${VAL_TFRECORD}"
                    echo "IMAGE_HEIGHT : ${IMAGE_HEIGHT}"
                    echo "IMAGE_WIDTH : ${IMAGE_WIDTH}"
                    echo "CHANNELS : ${CHANNELS}"
                }
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