import os

# Directories
OUTPUT_DIR = "/scratch/m23csa017/ViT/ViT/output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_DIR = "/scratch/m23csa017/ViT/ViT/data" 

# Hyperparameters
PATCH_SIZE = 16
BATCH_SIZE = 128  
IMG_SIZE = 224  
EMBEDDING_DIM = 768
MLP_SIZE = 3072
NUM_HEADS = 12
NUM_EPOCHS = 20  
NUM_WORKERS = 4  
LR = 1e-4
