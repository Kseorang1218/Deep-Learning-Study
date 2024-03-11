from torchvision import transforms
import numpy as np
import torchvision.datasets as datasets
from typing import Tuple

def get_mean_std(
        train_data: datasets, test_data: datasets
        ) -> Tuple[list, list, list, list]:
    """
    데이터셋의 mean, std를 구하는 함수

    Parameters
    ---------- 
    train_data: torchvision.datasets
        train dataset
    test_data: torchvision.datasets
        test dataset

    Returns
    ----------
    Tuple[list, list, list, list]
        튜플 (train_mean, train_std, test_mean, test_std)
        
    Examples
    ----------
    >>> train_mean, train_std, test_mean, test_std = get_mean_std(train_dataset, test_dataset)
    """
    train_mean_rgb = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_data]
    train_std_rgb = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_data]

    train_mean_r = np.mean([m[0] for m in train_mean_rgb])
    train_mean_g = np.mean([m[1] for m in train_mean_rgb])
    train_mean_b = np.mean([m[2] for m in train_mean_rgb])

    train_std_r = np.mean([s[0] for s in train_std_rgb])
    train_std_g = np.mean([s[1] for s in train_std_rgb])
    train_std_b = np.mean([s[2] for s in train_std_rgb])

    test_mean_rgb = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in test_data]
    test_std_rgb = [np.std(x.numpy(), axis=(1, 2)) for x, _ in test_data]

    test_mean_r = np.mean([m[0] for m in test_mean_rgb])
    test_mean_g = np.mean([m[1] for m in test_mean_rgb])
    test_mean_b = np.mean([m[2] for m in test_mean_rgb])

    test_std_r = np.mean([s[0] for s in test_std_rgb])
    test_std_g = np.mean([s[1] for s in test_std_rgb])
    test_std_b = np.mean([s[2] for s in test_std_rgb])

    train_mean = [train_mean_r, train_mean_g, train_mean_b]
    train_std = [train_std_r, train_std_g, train_std_b]
    test_mean = [test_mean_r, test_mean_g, test_mean_b]
    test_std = [test_std_r, test_std_g, test_std_b]

    return train_mean, train_std, test_mean, test_std

def do_transform(
        train_mean: list, train_std: list, test_mean: list, test_std: list
        ) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    train, test 데이터셋의 transform을 하는 함수

    Parameters
    ---------- 
    train_mean: list
        train dataset의 mean
    train_std: list
        train dataset의 std
    test_mean: list
        test dataset의 mean
    test_std: list
        test dataset의 std

    Returns
    ----------
    Tuple[transforms.Compose, transforms.Compose]
        튜플 (train_transform, test_transform)
        
    Examples
    ----------
    >>> train_transform, test_transform = do_transform(train_mean, train_std, test_mean, test_std)
        train_dataset.transform = train_transform
        test_dataset.transform = test_transform
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([train_mean[0], train_mean[1], train_mean[2]], [train_std[0], train_std[1], train_std[2]]), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomAffine(
                degrees=15, translate=(0.2, 0.2),
                scale=(0.8, 1.2), shear=15)]),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([test_mean[0], test_mean[1], test_mean[2]], [test_std[0], test_std[1], test_std[2]]),
        ])

    return train_transform, test_transform
