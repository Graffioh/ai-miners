# Experiments

## Accuracy (simple K-Fold validation)

- K = 5
- BATCH_SIZE = 128
- EPOCHS = 5
- Optimizer: Adam
- Loss: CrossEntropyLoss
- LEARNING_RATE: 0.001

### FCN

- **Normal** (random, shit)
    - dropout 0.3
    - Mean Validation Accuracy: 12.37% ± 0.09%
    - Min Validation Accuracy: 12.28%
    - Max Validation Accuracy: 12.54%
    - Individual Fold Accuracies: ['12.54%', '12.35%', '12.39%', '12.32%', '12.28%']
    - EXTRA: 1 direction with 100% whereas others are 0%, randomly, example: 
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

- **FCN with Batch Norm** (poor generalization)
    - dropout 0.5 
    - Mean Validation Accuracy: 35.92% ± 13.86%
    - Min Validation Accuracy: 19.65%
    - Max Validation Accuracy: 59.88%
    - Individual Fold Accuracies: ['36.47%', '38.41%', '59.88%', '19.65%', '25.17%']
    - Test Accuracy: 61.00%

- **FCN with Batch Norm** (good generalization but poor performances)
    - dropout 0.3
    - Mean Validation Accuracy: 39.48% ± 10.09%
    - Min Validation Accuracy: 26.95%
    - Max Validation Accuracy: 52.29%
    - Individual Fold Accuracies: ['26.95%', '30.61%', '49.80%', '37.78%', '52.29%']
    - Test Accuracy: 40.08%


### CNN

- **Normal** (good but overfitting validation)
    - MaxPool2d(2,2)
    - dropout 0.5
    - Mean Validation Accuracy: 90.45% ± 0.09%
    - Min Validation Accuracy: 90.29%
    - Max Validation Accuracy: 90.57%
    - Individual Fold Accuracies: ['90.29%', '90.52%', '90.57%', '90.44%', '90.45%']
    - Test Accuracy: 78.46%

- **Normal** (a bit better but still overfitting)
    - MaxPool2d(2,2)
    - dropout 0.3
    - Mean Validation Accuracy: 91.69% ± 0.15%
    - Min Validation Accuracy: 91.45%
    - Max Validation Accuracy: 91.89%
    - Individual Fold Accuracies: ['91.45%', '91.64%', '91.89%', '91.73%', '91.76%']
    - Test Accuracy: 79.73%

- **Normal** (bad)
    - MaxPool2d(3,3) with AdaptiveAvgPool2d(4,4)
    - dropout 0.3
    - Mean Validation Accuracy: 83.35% ± 1.06%
    - Min Validation Accuracy: 81.65%
    - Max Validation Accuracy: 84.67%
    - Individual Fold Accuracies: ['81.65%', '84.11%', '82.77%', '83.54%', '84.67%']
    - Test Accuracy: 68.16%

- **Normal** (more balanced than the others)
    - MaxPool2d(3,2,1) with AdaptiveAvgPool2d(2,2)
    - dropout 0.3
    - Mean Validation Accuracy: 84.30% ± 0.54%
    - Min Validation Accuracy: 83.50%
    - Max Validation Accuracy: 84.93%
    - Individual Fold Accuracies: ['83.92%', '84.81%', '84.93%', '84.32%', '83.50%']
    - Test Accuracy: 71.76%

- **Normal** ()
    - AdamW
    - MaxPool2d(2,2)
    - dropout 0.5
    - Mean Validation Accuracy: 91.33% ± 0.16%
    - Min Validation Accuracy: 91.09%
    - Max Validation Accuracy: 91.59%
    - Individual Fold Accuracies: ['91.39%', '91.26%', '91.59%', '91.09%', '91.33%']
    - Test Accuracy: 79.18%

- **with Batch Norm everywhere** (shit)
    - MaxPool2d(2,2)
    - dropout 0.5
    - Mean Validation Accuracy: 64.25% ± 18.90%
    - Min Validation Accuracy: 40.72%
    - Max Validation Accuracy: 86.55%
    - Individual Fold Accuracies: ['75.44%', '75.99%', '86.55%', '42.52%', '40.72%']
    - Test Accuracy: 44.26%

- **with Batch Norm everywhere** (shittier)
    - MaxPool2d(2,2)
    - dropout 0.3
    - Mean Validation Accuracy: 65.53% ± 24.51%
    - Min Validation Accuracy: 25.94%
    - Max Validation Accuracy: 89.54%
    - Individual Fold Accuracies: ['89.54%', '77.67%', '86.03%', '25.94%', '48.46%']
    - Test Accuracy: 27.42%

- **with Batch Norm NOT in the FCN**
    - MaxPool2d(2,2)
    - dropout 0.5
    - Mean Validation Accuracy: 87.06% ± 1.01%
    - Min Validation Accuracy: 85.83%
    - Max Validation Accuracy: 88.52%
    - Individual Fold Accuracies: ['88.52%', '87.89%', '86.20%', '85.83%', '86.85%']
    - Test Accuracy: 74.15%

- **with Batch Norm ONLY in the FCN**
    - MaxPool2d(2,2)
    - dropout 0.5
    - Mean Validation Accuracy: 89.97% ± 0.31%
    - Min Validation Accuracy: 89.51%
    - Max Validation Accuracy: 90.42%
    - Individual Fold Accuracies: ['89.51%', '89.92%', '90.18%', '89.83%', '90.42%']
    - Test Accuracy: 77.28%

- **less aggressive pooling**

- **without pooling**


### Improved CNN (not that improved)

- Batch norm
- MaxPool2d(kernel_size=3, stride=2, padding=1)
- AdaptiveAvgPool2d((2, 2))
- dropout 0.3
- Mean Validation Accuracy: 69.56% ± 9.35%
- Min Validation Accuracy: 52.32%
- Max Validation Accuracy: 80.56%
- Individual Fold Accuracies: ['80.56%', '70.69%', '73.66%', '52.32%', '70.56%']
- Test Accuracy: 64.10%

### ResNet

...

## Logging

*for detailed logs go into sprites/runs folder after training and running the model*

## TODO

### Furlanello meeting 1 

- [] arrivare a 86% tramite CNN base
- [] improve plotting
- [] confusion matrix
- [] separare accuracy per direzione e per sprite
- [] se sprite accuracy bassa, fare confusion matrix per quelle sprite (analisi approfondita)
- [] testare più volte su template diversi per esperimenti (purged k fold)
- [V] togliere k fold 
- [V] gestire in 3 canali non 4 con trasparenza
- [] downsamplare di meno

- differenza tra modello trainato tra 5 e 10 epoch
- se traino modello piu volte, cambia la confusion matrix?
- data augmentation (cambiare colore sfondi, traslare rispetto allo sfondo l'immagine per la pos) / capire trasparenza
- weight decay da adam = dropout inutile (?)

- vedere codici di torch vision
- https://d2l.ai/chapter_computer-vision/fcn.html


