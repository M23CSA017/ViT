from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from pathlib import Path
from data_preprocessing import get_transforms

def get_data_loaders(batch_size=128, num_workers=4):
    # Use the paths you specified
    test_dir = Path("/scratch/m23csa017/ViT/ViT/data/test_data/food-101/images")
    test_dir = Path("/scratch/m23csa017/ViT/ViT/data/train_data/food-101/images")
    

    transforms_train, transforms_test = get_transforms()

    # Function to check if dataset is downloaded
    def is_dataset_downloaded(directory):
        # Check for the presence of a specific file or directory to confirm download
        return (directory / "images").exists() and (directory / "meta").exists()

    # Download dataset only if not already present
    if not is_dataset_downloaded(train_dir):
        print("Downloading training data...")
        # train_data = Food101(root=train_dir, split="train", transform=transforms_train, download=True)
    else:
        print("Training data already downloaded.")
        train_data = Food101(root=train_dir, split="train", transform=transforms_train, download=False)

    if not is_dataset_downloaded(test_dir):
        print("Downloading testing data...")
        # test_data = Food101(root=test_dir, split="test", transform=transforms_test, download=True)
    else:
        print("Testing data already downloaded.")
        test_data = Food101(root=test_dir, split="test", transform=transforms_test, download=False)

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Data loaders created.")
    print(len(train_dataloader))
    print(len(test_dataloader)) 
    return train_dataloader, test_dataloader
