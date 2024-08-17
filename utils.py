import torch
from pathlib import Path
import matplotlib.pyplot as plt
import os

def save_model(model, target_dir, model_name):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith((".pth", ".pt")), "Model name should end with .pth or .pt"
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def unnormalize_image(image_tensor, mean, std):
    """Undo normalization on an image tensor."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image_tensor = image_tensor * std + mean
    return torch.clamp(image_tensor, 0, 1)

def ensure_dir_exists(file_path):
    """Ensure the directory for the file path exists."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_image(image, label, class_names, save_path):
    """Plot an image with its label and save to file."""
    ensure_dir_exists(save_path)
    image = unnormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.permute(1, 2, 0)  # Convert tensor to HWC format
    plt.imshow(image)
    plt.title(class_names[label])
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def plot_patches(image, patch_size, save_path):
    """Plot image patches in a grid format and save to file."""
    ensure_dir_exists(save_path)
    image = unnormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_size = image.shape[1]
    num_patches = img_size // patch_size

    fig, axis = plt.subplots(nrows=num_patches, ncols=num_patches, figsize=(num_patches, num_patches), sharex=True, sharey=True)

    for i, patch_height in enumerate(range(0, img_size, patch_size)):
        for j, patch_width in enumerate(range(0, img_size, patch_size)):
            patch = image[:, patch_height:patch_height + patch_size, patch_width:patch_width + patch_size].permute(1, 2, 0)
            axis[i, j].imshow(patch)
            axis[i, j].set_xticks([])
            axis[i, j].set_yticks([])

    fig.suptitle("Patchified Image", fontsize=14)
    plt.savefig(save_path)
    plt.close()

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    """Plot loss and accuracy curves for training and validation and save to file."""
    ensure_dir_exists(save_path)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_path)
    plt.close()
