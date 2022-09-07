# Code insipired by: https://tinyurl.com/tiny-imagenet
from pathlib import Path
import shutil
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

URL_DATA = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


def get_data(data_dir=None,
             val_split=0.2,
             transforms=None,
             test_transforms=None
             ) -> Tuple[Dataset, ...]:
    if data_dir is None:
        data_dir = Path.cwd() / 'data'
    else:
        data_dir = Path(data_dir)
    if not data_dir.exists():  # Check existence
        data_dir.mkdir(parents=True)
    # TODO: Use checksum to verify if we need to download data
    if next(data_dir.iterdir(), None) is None:  # Check if empty
        # Download data
        download_and_extract_archive(URL_DATA, str(data_dir))
        (data_dir / 'tiny-imagenet-200.zip').unlink()

    train_data_dir = str(data_dir / 'tiny-imagenet-200' / 'train')
    # Train data are already subdivided in (image, label) format
    if transforms is None:
        ds_train_val = ImageFolder(train_data_dir, transform=ToTensor())
    else:
        ds_train_val = ImageFolder(train_data_dir, transform=transforms)

    # Split train_val data in training and validation
    val_len = int(val_split * len(ds_train_val))
    train_len = len(ds_train_val) - val_len
    ds_train, ds_val = random_split(ds_train_val, [train_len, val_len])

    # Validation data are here used as Test data
    test_data_dir = data_dir / 'tiny-imagenet-200' / 'val'
    if (test_data_dir / 'val_annotations.txt').exists():
        # Validation data folder need to be organized in labels
        # First two columns of file 'tiny-imagenet-200/val/val_annotations.txt'
        # contains img filename and label
        with open(test_data_dir / 'val_annotations.txt', 'r') as f:
            test_image_dict = dict()
            for line in f:
                words = line.split('\t')
                test_image_dict[words[0]] = words[1]
        for image, folder in test_image_dict.items():
            newpath = test_data_dir / folder
            newpath.mkdir(exist_ok=True)
            shutil.move(
                str(test_data_dir / 'images' / image),
                newpath)
        shutil.rmtree(test_data_dir / 'images')
        (test_data_dir / 'val_annotations.txt').unlink()

    if test_transforms is None:
        ds_test = ImageFolder(str(test_data_dir), transform=ToTensor())
    else:
        ds_test = ImageFolder(str(test_data_dir), transform=test_transforms)

    return ds_train, ds_val, ds_test


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size=64,
                      num_workers=2
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = datasets
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
