import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
import torch
from torch.utils.data import random_split
from dataset import PGLSDataset
from models import PGLSModel, EnsemblePGLSModel, PGLSModelLateFusion
from torchrec.models.deepfm import DenseArch
import argparse
import timm
from typing import Literal

from train import train_model


BATCH_SIZE = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_images_path = 'data/train_images'
    train_csv_path = 'data/train.csv'

    tabular_data = pd.read_csv(train_csv_path)
    targets = ["X4", "X11", "X18", "X26", "X50", "X3112"]

    # Filter data
    upper_values = {}
    for target in targets:
        upper_values[target] = tabular_data[target+"_mean"].quantile(0.99)
        tabular_data = tabular_data[tabular_data[target+"_mean"] < upper_values[target]]
        tabular_data = tabular_data[tabular_data[target+"_mean"] > 0]

    # Normalize the targets
    original_means = tabular_data[[f"{target}_mean" for target in targets]].mean()
    original_stds = tabular_data[[f"{target}_mean" for target in targets]].std()
    tabular_data[[f"{target}_mean" for target in targets]] = (tabular_data[[f"{target}_mean" for target in targets]] - original_means) / original_stds

    # Normalize the features
    tabular_input_size = 0
    for column in tabular_data.columns:
        if column in ["id"]+[target+"_mean" for target in targets]+[target+"_sd" for target in targets]:
            continue
        tabular_input_size += 1
        min_val = tabular_data[column].min()
        max_val = tabular_data[column].max()
        tabular_data[column] = (tabular_data[column] - min_val) / (max_val - min_val)

    train_images_dataset = ImageFolder(root=train_images_path, transform=transform)

    train_image_csv_dataset = PGLSDataset(tabular_data=tabular_data, image_folder=train_images_dataset, transform_csv=None)
    train, val = random_split(train_image_csv_dataset, [int(0.9*len(train_image_csv_dataset)), len(train_image_csv_dataset) - int(0.9*len(train_image_csv_dataset))])

    train_data_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    original_stds = torch.from_numpy(original_stds.values).float()
    original_means = torch.from_numpy(original_means.values).float()

    return train_data_loader, val_data_loader, original_means, original_stds, tabular_input_size


def prepare_model(model_type: Literal["midfusion", "ensemble", "latefustion"],
                  tabular_input_size: int):
    if model_type == "midfusion":
        effnet = timm.create_model(
            'efficientnet_b4.ra2_in1k',
            pretrained=True,
            num_classes=0,
        )
        model = PGLSModel(effnet, tabular_input_size)
    elif model_type == "ensemble":
        effnet = timm.create_model(
            'efficientnet_b4.ra2_in1k',
            pretrained=True,
            num_classes=0,
        )
        xception = timm.create_model('inception_resnet_v2.tf_in1k', pretrained=True, num_classes=0)
        densenet = timm.create_model('densenet121.ra_in1k', pretrained=True, num_classes=0)
        model = EnsemblePGLSModel([effnet, xception, densenet], tabular_input_size)
    elif model_type == "latefusion":
        BiT = timm.create_model(
            'resnetv2_101x1_bitm',
            pretrained=True,
            num_classes=0,
        )
        tabular_model = DenseArch(in_features=tabular_input_size, hidden_layer_size=tabular_input_size * 2, embedding_dim=tabular_input_size)
        model = PGLSModelLateFusion(BiT, tabular_model, tabular_input_size)
    else:
        raise ValueError(f"Model type {model_type} not recognized")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="midfusion", help="Type of model to train: midfusion, ensemble, latefusion")
    args = parser.parse_args()

    train_data_loader, val_data_loader, original_means, original_stds, tabular_input_size = prepare_data()
    model = prepare_model(args.model_type, tabular_input_size)
    train_model(model, train_data_loader, val_data_loader, args.model_type, original_means, original_stds, validation_after_n_batches=20)
