from data_preparation import (
    print_dataset_summary,
    show_example_per_class,
    get_dataloaders
)

from model import create_model, get_device
from train import train_model, plot_history
from evaluate import evaluate_model

DATASET_PATH = "../dataset_waste_container"
SAVE_PATH = "best_model.pth"

def main():

    print_dataset_summary(DATASET_PATH)
    show_example_per_class(DATASET_PATH)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATASET_PATH)

    device = get_device()
    model = create_model(num_classes=len(class_names), freeze_backbone=True)
    model.to(device)

    history = train_model(model, train_loader, val_loader)

    plot_history(history)

    evaluate_model(model, test_loader, class_names, device=device, model_path=SAVE_PATH)

if __name__ == "__main__":
    main()
