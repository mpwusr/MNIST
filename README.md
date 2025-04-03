# MNIST Digit Classification with PyTorch CNNs

This repository contains two Python scripts for classifying handwritten digits from the MNIST dataset using Convolutional Neural Networks (CNNs) implemented in PyTorch. The scripts demonstrate both standard training/testing and k-fold cross-validation approaches.

## Scripts
1. **`mnist_cnn.py`**  
   - Standard CNN training with train/test split
   - Source: Adapted from [Nextjournal PyTorch MNIST](https://nextjournal.com/gkoehler/pytorch-mnist)
2. **`mnist_cnn_kfold.py`**  
   - CNN training with k-fold cross-validation
   - Custom implementation with timing and weight reset

## Dataset
- **Source**: MNIST Handwritten Digit Dataset
- **Description**: 28x28 grayscale images of digits (0-9)
- **Size**: 60,000 training images, 10,000 test images
- **Preprocessing**: Normalized with mean=0.1307, std=0.3081

## Features
### `mnist_cnn.py`
- **Model**: CNN with 2 convolutional layers, dropout, and 2 fully connected layers
- **Training**: 
  - 10 epochs, SGD optimizer with momentum
  - Batch sizes: 64 (train), 1000 (test)
- **Visualization**: 
  - Sample images with ground truth and predictions
  - Training and test loss plots
- **Extras**: Demonstrates model/optimizer state saving and reloading

### `mnist_cnn_kfold.py`
- **Model**: Simpler CNN with 1 convolutional layer and 3 fully connected layers
- **Training**: 
  - 10-fold cross-validation, 1 epoch per fold
  - Adam optimizer, batch sizes: 100 (train), 10 (test)
- **Evaluation**: 
  - Accuracy per fold and average across folds
  - Training time measurement
- **Extras**: Weight resetting between folds to avoid leakage

## Requirements

## Usage
1. Run either script:
   ```bash
   python mnist_cnn.py
   ```
   ```bash
   python mnist_cnn_kfold.py
   ```
2. Outputs:
   - **mnist_cnn.py**: Console logs, loss plots, sample predictions, saved model/optimizer states (`model.pth`, `optimizer.pth`)
   - **mnist_cnn_kfold.py**: Console logs with fold accuracies, saved models per fold (`model-fold-X.pth`)

## Configuration
### `mnist_cnn.py`
- `n_epochs`: 10
- `batch_size_train`: 64
- `batch_size_test`: 1000
- `learning_rate`: 0.01
- `momentum`: 0.5

### `mnist_cnn_kfold.py`
- `k_folds`: 10
- `num_epochs`: 1
- `batch_size` (train/test): 100/10
- `learning_rate`: 1e-4

## Output
### `mnist_cnn.py`
- **Console**:
  ```
  Train Epoch: 1 [0/60000 (0%)]	Loss: 2.302585
  Test set: Avg. loss: 0.1234, Accuracy: 9750/10000 (98%)
  ```
- **Plots**: Loss curves, sample digit predictions

### `mnist_cnn_kfold.py`
- **Console**:
  ```
  FOLD 0
  Accuracy for fold 0: 95 %
  ...
  Average: 94.5 %
  It took 5.2 minutes to run...
  ```
- **Files**: Model states per fold

## Example Output
- **mnist_cnn.py**: Loss plot and predicted digits  
  *(Add `loss_plot.png` or `predictions.png` manually if desired)*
- **mnist_cnn_kfold.py**: Fold-wise accuracy summary

## Notes
- Both scripts download MNIST data to a `data/` directory automatically
- `mnist_cnn.py` includes an SSL workaround for downloading on some systems
- `mnist_cnn_kfold.py` resets weights to ensure independent fold training
- Adjust hyperparameters in the scripts for experimentation

## License
This project is open-source and available under the MIT License.
```

### Key Points
- **Overview**: Introduces the purpose and distinguishes the two scripts.
- **Scripts**: Lists both files with their sources and key differences.
- **Features**: Details the functionality of each script separately.
- **Requirements**: Unified package list for both scripts.
- **Usage**: Clear instructions for running each script.
- **Configuration**: Highlights key adjustable parameters.
- **Output**: Describes console and visual outputs with placeholders for images.
- **Notes**: Includes practical tips and dependencies.

Since two of the scripts were identical, I treated them as one (`mnist_cnn.py`) and contrasted it with the k-fold version (`mnist_cnn_kfold.py`). This README provides a comprehensive guide for users while avoiding redundancy. Let me know if you'd like to adjust anything or add specific output examples!
