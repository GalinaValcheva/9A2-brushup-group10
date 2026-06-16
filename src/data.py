import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

import medmnist
from medmnist import INFO


DATA_FLAG = "pneumoniamnist"


def get_data_class():
    info = INFO[DATA_FLAG]
    data_class = getattr(medmnist, info["python_class"])
    return info, data_class


def get_torch_loaders(batch_size: int = 64):
    info, data_class = get_data_class()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = data_class(split="train", transform=transform, download=True)
    val_dataset = data_class(split="val", transform=transform, download=True)
    test_dataset = data_class(split="test", transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def get_flattened_data():
    info, data_class = get_data_class()

    train_dataset = data_class(split="train", download=True)
    val_dataset = data_class(split="val", download=True)
    test_dataset = data_class(split="test", download=True)

    x_train = train_dataset.imgs.reshape(len(train_dataset), -1).astype(np.float32) / 255.0
    x_val = val_dataset.imgs.reshape(len(val_dataset), -1).astype(np.float32) / 255.0
    x_test = test_dataset.imgs.reshape(len(test_dataset), -1).astype(np.float32) / 255.0

    y_train = train_dataset.labels.squeeze().astype(int)
    y_val = val_dataset.labels.squeeze().astype(int)
    y_test = test_dataset.labels.squeeze().astype(int)

    return x_train, x_val, x_test, y_train, y_val, y_test
