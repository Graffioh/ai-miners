# Experiments

## Accuracy (K-Fold validation)

### FCN

- **Normal**
    - dropout 0.3
    - LEARNING_RATE: 0.001
    - BATCH_SIZE: 128
    - EPOCHS: 5
    - Mean Validation Accuracy: 12.37% ± 0.09%
    - Min Validation Accuracy: 12.28%
    - Max Validation Accuracy: 12.54%
    - Individual Fold Accuracies: ['12.54%', '12.35%', '12.39%', '12.32%', '12.28%']
    - **EXTRA**: 1 direction with 100% whereas others are 0%, randomly, example: 
        *Fold 4*
        Training samples: 87552
        Validation samples: 21888
        Validation Accuracy: 12.32%
        Direction Accuracies:
            - East: 0.00%
            - NE: 0.00%
            - NW: 0.00%
            - North: 100.00%
            - SE: 0.00%
            - SW: 0.00%
            - South: 0.00%
            - West: 0.00%
    - Test Accuracy: 12.40%

- **FCN with Batch Norm** 
    - dropout 0.5 
    - LEARNING_RATE: 0.001
    - BATCH_SIZE: 128
    - EPOCHS: 5
    - Mean Validation Accuracy: 35.92% ± 13.86%
    - Min Validation Accuracy: 19.65%
    - Max Validation Accuracy: 59.88%
    - Individual Fold Accuracies: ['36.47%', '38.41%', '59.88%', '19.65%', '25.17%']
    - Test Accuracy: 61.00%

- **FCN with Batch Norm** 
    - dropout 0.3
    - LEARNING_RATE: 0.001
    - BATCH_SIZE: 128
    - EPOCHS: 5
    - Mean Validation Accuracy: 39.48% ± 10.09%
    - Min Validation Accuracy: 26.95%
    - Max Validation Accuracy: 52.29%
    - Individual Fold Accuracies: ['26.95%', '30.61%', '49.80%', '37.78%', '52.29%']
    - Test Accuracy: 40.08%


### CNN

- **Normal** 
    - dropout 0.3
    - LEARNING_RATE: 0.001
    - BATCH_SIZE: 128
    - EPOCHS: 5
    - Mean Validation Accuracy: 90.45% ± 0.09%
    - Min Validation Accuracy: 90.29%
    - Max Validation Accuracy: 90.57%
    - Individual Fold Accuracies: ['90.29%', '90.52%', '90.57%', '90.44%', '90.45%']
    - Test Accuracy: 78.46%

- **with Batch Norm everywhere** 
    - dropout 0.5
    - LEARNING_RATE: 0.001
    - BATCH_SIZE: 128
    - EPOCHS: 5
    - Mean Validation Accuracy: 64.25% ± 18.90%
    - Min Validation Accuracy: 40.72%
    - Max Validation Accuracy: 86.55%
    - Individual Fold Accuracies: ['75.44%', '75.99%', '86.55%', '42.52%', '40.72%']
    - Test Accuracy: 44.26%

- **with Batch Norm everywhere** 
    - dropout 0.3
    - LEARNING_RATE: 0.001
    - BATCH_SIZE: 128
    - EPOCHS: 5
    - Mean Validation Accuracy: 65.53% ± 24.51%
    - Min Validation Accuracy: 25.94%
    - Max Validation Accuracy: 89.54%
    - Individual Fold Accuracies: ['89.54%', '77.67%', '86.03%', '25.94%', '48.46%']
    - Test Accuracy: 27.42%

- **with Batch Norm not in the FCN classification part**

- **less aggressive pooling**

- **without pooling**


### ResNet

...

## Logging

*for detailed logs go into sprites/runs folder after training and running the model*

## TODO

- improve plotting
- mini batch stratified sampling