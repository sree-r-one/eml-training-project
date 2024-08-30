pipeline{
    agent any 

    parameters {
        string(name: 'EPOCHS', defaultValue: '300', description: 'Number of epochs to train')
        string(name: 'BATCH_SIZE', defaultValue: '8', description: 'Batch size')
        string(name: 'LEARNING_RATE', defaultValue: '0.0001', description: 'Learning rate')
        string(name: 'TRAIN_TFRECORD', defaultValue: 'D:/LS3_LPC/Data/Train/train.tfrecords', description: 'Path to training TFRecord')
        string(name: 'VAL_TFRECORD', defaultValue: 'D:/LS3_LPC/Data/Val/val.tfrecords', description: 'Path to validation TFRecord')
    }

    stages{
        stage('Parameters'){
            steps{
                script{
                    echo "EPOCHS : ${params.EPOCHS}"
                    echo "BATCH_SIZE : ${params.BATCH_SIZE}"
                    echo "LEARNING_RATE : ${params.LEARNING_RATE}"
                    echo "TRAIN_TFRECORD : ${params.TRAIN_TFRECORD}"
                    echo "VAL_TFRECORD : ${params.VAL_TFRECORD}"
                }
            }
        }
    }


}