<<<<<<< HEAD
# Vision Transformer Project

## Overview
This project implements a Vision Transformer model for image classification using the Food101 dataset. The model is designed to handle image classification tasks and is built using PyTorch.

## Structure
- `data_preprocessing.py`: Data transformation functions.
- `data_loaders.py`: Dataset loading and DataLoader creation.
- `model.py`: Vision Transformer model definition.
- `utils.py`: Utility functions for metrics and other tasks.
- `visualize.py`: Visualization functions for images and patches.
- `main.py`: Main script to run the project.
- `train.py`: Training functions for train and validation.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>

    cd VisionTransformerProject
    pip install -r requirements.txt



## Usage
1. Train the Model:

- You can start the training process by running the train.py script. Adjust parameters such as batch size and learning rate as needed.
    - `python train.py --batch_size 32 --epochs 10 --learning_rate 0.001`

- To execute the main script, which includes data loading, training, and evaluation, use:
    - `python main.py`

- To visualize results such as images, patches, and training metrics, use the visualize.py script:
    - `python visualize.py`

## Configuration
- You can adjust the following configurations in the main.py and train.py files:

- Learning Rate: Adjust in the optimizer settings.
- Number of Epochs: Set in the training script.
- Batch Size: Configurable in the data_loaders.py script.


## Example
To train the model with specific parameters, you can use:
- `python train.py --batch_size 32 --epochs 10 --learning_rate 0.001`
=======
# ViT
>>>>>>> origin/main
