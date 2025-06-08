# Experiments

## Accuracy (K-Fold validation)

### FCN

- **Normal**
    - dropout 0.3
    - LEARNING_RATE: 0.001
    - BATCH_SIZE: 128
    - EPOCHS: 5
    - VALIDATION_SPLIT_RATIO: 0.2
    - Mean Validation Accuracy: 12.37% ± 0.09%
    - Min Validation Accuracy: 12.28%
    - Max Validation Accuracy: 12.54%
    - Individual Fold Accuracies: ['12.54%', '12.35%', '12.39%', '12.32%', '12.28%']
    - **EXTRA**: 100% on only 1 direction while the others 0%, randomly, example: 
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

- **FCN with Batch Norm (dropout 0.3)** 

- **FCN with Batch Norm (dropout 0.5)** 


### CNN

- **Normal** 

- **with Batch Norm everywhere** 
    - dropout 0.5
    - LEARNING_RATE: 0.001
    - BATCH_SIZE: 128
    - EPOCHS: 5
    - VALIDATION_SPLIT_RATIO: 0.2
    - Mean Validation Accuracy: 64.25% ± 18.90%
    - Min Validation Accuracy: 40.72%
    - Max Validation Accuracy: 86.55%
    - Individual Fold Accuracies: ['75.44%', '75.99%', '86.55%', '42.52%', '40.72%']

- **with Batch Norm everywhere** 
    - dropout 0.3
    - LEARNING_RATE: 0.001
    - BATCH_SIZE: 128
    - EPOCHS: 5
    - Mean Validation Accuracy: 65.53% ± 24.51%
    - Min Validation Accuracy: 25.94%
    - Max Validation Accuracy: 89.54%
    - Individual Fold Accuracies: ['89.54%', '77.67%', '86.03%', '25.94%', '48.46%']

- **with Batch Norm not in the FCN classification part**

- **less aggressive pooling**

- **without pooling**


### ResNet

## Logging

*for detailed logs go into sprites/runs folder after training and running the model*

## TODO

- fix batchnorm mean and variance memorization after k-fold (during final training, we should keep the best normalization from k-folds?)
- improve plotting
- mini batch stratified sampling