import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from pathlib import Path
import config

# Define transformations
food101_transforms_train = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256 before cropping
    transforms.RandomCrop((config.IMG_SIZE, config.IMG_SIZE)),  # Random crop to IMG_SIZE
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

food101_transforms_test = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),  # Direct resize to IMG_SIZE
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data():
    data_dir = Path(config.DATA_DIR)
    train_dir = data_dir / "train_data/food-101/images"
    test_dir = data_dir / "test_data/food-101/images"

    # Load datasets
    train_data = datasets.ImageFolder(root=train_dir, transform=food101_transforms_train)
    test_data = datasets.ImageFolder(root=test_dir, transform=food101_transforms_test)

    # DataLoaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataloader = DataLoader(dataset=test_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    return train_dataloader, val_dataloader, len(train_data.classes), train_data.classes
