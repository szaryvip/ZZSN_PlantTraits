import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
import torch
import argparse
from typing import Literal
from dataset import PGLSDataset

from train import denormalize_targets
from prepare_model import prepare_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_predictions(predictions, test_csv_dataframe, filename):
    with open(filename, "w") as f:
        f.write("id,X4,X11,X18,X26,X50,X3112\n")
        for pred, id in zip(predictions, test_csv_dataframe["id"]):
            pred = [p.item() for p in pred]
            f.write(f"{id},{','.join([str(p) for p in pred])}\n")


def prepare_data():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    batch_size = 256

    test_images_path = "data/test_images"

    train_csv_path = "data/train.csv"
    test_csv_path = "data/test.csv"

    tabular_data = pd.read_csv(train_csv_path)
    targets = ["X4", "X11", "X18", "X26", "X50", "X3112"]

    # Filter data
    upper_values = {}
    for target in targets:
        upper_values[target] = tabular_data[target + "_mean"].quantile(0.99)
        tabular_data = tabular_data[tabular_data[target + "_mean"] < upper_values[target]]
        tabular_data = tabular_data[tabular_data[target + "_mean"] > 0]

    original_means = tabular_data[[f"{target}_mean" for target in targets]].mean()
    original_stds = tabular_data[[f"{target}_mean" for target in targets]].std()

    tabular_input_size = 0

    test_tabular_data = pd.read_csv(test_csv_path)
    # Normalize the features
    for column in test_tabular_data.columns:
        if column in ["id"]:
            continue
        tabular_input_size += 1
        min_val = test_tabular_data[column].min()
        max_val = test_tabular_data[column].max()
        test_tabular_data[column] = (test_tabular_data[column] - min_val) / (
            max_val - min_val
        )

    test_images_dataset = ImageFolder(root=test_images_path,
                                      transform=transform)

    test_image_csv_dataset = PGLSDataset(
        tabular_data=test_tabular_data, image_folder=test_images_dataset,
        transform_csv=None
    )

    test_data_loader = DataLoader(
        test_image_csv_dataset, batch_size=batch_size, shuffle=False
    )

    original_stds = torch.from_numpy(original_stds.values).float()
    original_means = torch.from_numpy(original_means.values).float()

    return test_data_loader, original_means, original_stds, tabular_input_size, test_tabular_data


def load_model(model_type: Literal["midfusion", "ensemble", "latefustion"],
               tabular_input_size):
    model = prepare_model(model_type, tabular_input_size)
    model.load_state_dict(torch.load(f"models/{model_type}_best_epoch.pt"))
    return model

def make_predictions(model, test_data_loader, test_tabular_data,
                     original_means, original_stds):
    predictions = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in test_data_loader:
            image, features, targets = data
            image = image.to(device)
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(image, features)
            outputs = outputs.to("cpu")
            targets = targets.to("cpu")
            outputs_denorm = denormalize_targets(outputs, original_means,
                                                 original_stds)
            predictions.append(outputs_denorm)
    predictions = [item for sublist in predictions for item in sublist]

    save_predictions(predictions, test_tabular_data, "predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="midfusion", help="Type of model to train: midfusion, ensemble, latefusion")
    args = parser.parse_args()

    test_data_loader, original_means, original_stds, tabular_input_size, test_tabular_data = prepare_data()
    model = load_model(args.model_type, tabular_input_size)
    make_predictions(model, test_data_loader, test_tabular_data,
                     original_means, original_stds)
