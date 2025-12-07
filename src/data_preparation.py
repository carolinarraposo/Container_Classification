import os
import random
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter

# =====================================================
# 0. DATA EXPLORATION
# =====================================================

def get_class_names(dataset_path):
    return sorted([
        c for c in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, c))
    ])


def count_images_per_class(dataset_path):
    counts = {}
    for cls in get_class_names(dataset_path):
        cls_path = os.path.join(dataset_path, cls)
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(("jpg", "jpeg", "png"))
        ]
        counts[cls] = len(images)
    return counts


def print_dataset_summary(dataset_path):
    print("\n=== DATASET SUMMARY ===\n")
    counts = count_images_per_class(dataset_path)
    for cls, n in counts.items():
        print(f"{cls}: {n} imagens")
    print("\n=======================\n")


def show_example_per_class(dataset_path):
    class_names = get_class_names(dataset_path)
    plt.figure(figsize=(4 * len(class_names), 4))

    for i, cls in enumerate(class_names):
        cls_path = os.path.join(dataset_path, cls)
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(("jpg", "jpeg", "png"))
        ]
        img_path = os.path.join(cls_path, random.choice(images))
        img = Image.open(img_path).convert("RGB")

        plt.subplot(1, len(class_names), i + 1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =====================================================
# 1. TRANSFORM APLICADOS A CADA CONJUNTO
# =====================================================

def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def get_eval_transforms(img_size=224):
    """Transform para validação E teste (sem augmentações)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# =====================================================
# 2. DATALOADERS COM DIVISÃO CORRETA DE DATASETS
# =====================================================

def get_dataloaders(
    dataset_path,
    img_size=224,
    batch_size=32,
    val_split=0.2,
    test_split=0.1
):

    # Dataset SEM transform — transforms são aplicados depois ao subset correto
    base_dataset = datasets.ImageFolder(dataset_path)

    class_names = base_dataset.classes
    total = len(base_dataset)

    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    # Divide indices, não datasets
    train_indices, val_indices, test_indices = random_split(
        range(total), [train_size, val_size, test_size]
    )

    # Cria datasets separados, cada um com o seu transform
    train_dataset = datasets.ImageFolder(dataset_path, transform=get_train_transforms(img_size))
    val_dataset   = datasets.ImageFolder(dataset_path, transform=get_eval_transforms(img_size))
    test_dataset  = datasets.ImageFolder(dataset_path, transform=get_eval_transforms(img_size))

    # Aplica só os índices correspondentes
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(val_dataset,   val_indices)
    test_dataset  = torch.utils.data.Subset(test_dataset,  test_indices)

    # ---------------------------
    # BALANCED SAMPLER (opcional mas útil)
    # ---------------------------
    train_labels = [base_dataset.targets[i] for i in train_indices]
    counts = Counter(train_labels)
    weights = [1.0 / counts[l] for l in train_labels]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    # LOADERS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names