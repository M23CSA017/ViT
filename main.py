import torch
import torch.optim as optim
import torch.nn as nn
from data_preprocessing import load_data
from model import ViT
from train import train
from utils import *
import config
from pathlib import Path
import time 

def main():
    # Load data
    torch.cuda.empty_cache()
    
    print(f"PyTorch version: {torch.__version__}")
    train_dataloader, val_dataloader, num_classes, class_names = load_data()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model setup
    model = ViT(
        img_size=config.IMG_SIZE,
        in_channels=3,
        patch_size=config.PATCH_SIZE,
        num_transformer_layers=12,
        embedding_dim=config.EMBEDDING_DIM,
        mlp_size=config.MLP_SIZE,
        num_heads=config.NUM_HEADS,
        num_classes=num_classes
    ).to(device)
    
    # Check model dtype and device
    print(f"Model dtype: {next(model.parameters()).dtype}, device: {next(model.parameters()).device}")
    
    # Visualize a sample image and its patches
    for images, labels in train_dataloader:
        image = images[0]
        label = labels[0]
        
        # Check image dtype and device
        print(f"Image dtype: {image.dtype}, device: {image.device}")
        
        # Save image with label
        plot_image(image, label, class_names, "output/sample_image.png")
        
        # Save patches of the image
        plot_patches(image, config.PATCH_SIZE, "output/image_patches.png")
        break  # Stop after the first batch for visualization
    
    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    # Create output directory for plots
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    # Train the model
    results = train(
        model=model,
        epochs=config.NUM_EPOCHS,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        accumulation_steps=4  # Include this if you're using gradient accumulation
    )
    
    # Save the model
    torch.save(model.state_dict(), output_dir / "vit_model.pth")
    
    # Plot and save loss and accuracy curves
    plot_loss_accuracy(
        results['train_loss'], 
        results['test_loss'], 
        results['train_acc'], 
        results['test_acc'],
        output_dir / "loss_accuracy_plot.png"
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total time taken: {minutes} minutes and {seconds} seconds")
    

    # Print final results
    print("Training complete.")
    print(f"Final Train Accuracy: {results['train_acc'][-1]:.4f}")
    print(f"Final Validation Accuracy: {results['test_acc'][-1]:.4f}")

if __name__ == "__main__":
    main()
