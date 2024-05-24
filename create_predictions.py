import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from torcheval.metrics import R2Score
from torchmetrics.functional import r2_score
import pandas as pd
import torch
from torch.utils.data import random_split
import shutil
import os
from dataset import PGLSDataset
from models import PGLSModel, EnsemblePGLSModel
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.empty_cache()

for folder in ["data/train_images", "data/test_images"]:
    if "0" not in os.listdir(folder):
        print("Moving images to 0 folder")
        os.makedirs(f"{folder}/0")
        for filename in os.listdir(folder):
            if filename.lower().endswith(".jpeg"):
                source_path = os.path.join(folder, filename)
                target_path = os.path.join(f"{folder}/0", filename)

                shutil.move(source_path, target_path)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert PIL image to tensor (H x W x C) in the range [0.0, 1.0]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize the image
    ]
)

batch_size = 256  # use 4 if problem with GPU memory


train_images_path = "data/train_images"
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

# Normalize the targets
original_means = tabular_data[[f"{target}_mean" for target in targets]].mean()
original_stds = tabular_data[[f"{target}_mean" for target in targets]].std()
tabular_data[[f"{target}_mean" for target in targets]] = (
    tabular_data[[f"{target}_mean" for target in targets]] - original_means
) / original_stds

# Normalize the features
tabular_input_size = 0
for column in tabular_data.columns:
    if column in ["id"] + [target + "_mean" for target in targets] + [
        target + "_sd" for target in targets
    ]:
        continue
    tabular_input_size += 1
    min_val = tabular_data[column].min()
    max_val = tabular_data[column].max()
    tabular_data[column] = (tabular_data[column] - min_val) / (max_val - min_val)


test_tabular_data = pd.read_csv(test_csv_path)
# Normalize the features
for column in test_tabular_data.columns:
    if column in ["id"]:
        continue
    min_val = test_tabular_data[column].min()
    max_val = test_tabular_data[column].max()
    test_tabular_data[column] = (test_tabular_data[column] - min_val) / (
        max_val - min_val
    )

train_images_dataset = ImageFolder(root=train_images_path, transform=transform)
test_images_dataset = ImageFolder(root=test_images_path, transform=transform)

train_image_csv_dataset = PGLSDataset(
    tabular_data=tabular_data, image_folder=train_images_dataset, transform_csv=None
)
train, val = random_split(
    train_image_csv_dataset,
    [
        int(0.8 * len(train_image_csv_dataset)),
        len(train_image_csv_dataset) - int(0.8 * len(train_image_csv_dataset)),
    ],
)
test_image_csv_dataset = PGLSDataset(
    tabular_data=test_tabular_data, image_folder=test_images_dataset, transform_csv=None
)


train_data_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(
    test_image_csv_dataset, batch_size=batch_size, shuffle=False
)

original_stds = torch.from_numpy(original_stds.values).float()
original_means = torch.from_numpy(original_means.values).float()


def denormalize_targets(targets):
    return targets * original_stds + original_means


# effnet = efficientnet_b0(weights=EfficientNet_B0_Weights)
effnet = timm.create_model(
    "efficientnet_b0.ra_in1k",
    pretrained=True,
    num_classes=0,
)

model = PGLSModel(effnet, tabular_input_size)
metric = R2Score()
criterion = torch.nn.MSELoss()

torch.cuda.empty_cache()


model.load_state_dict(torch.load("models/effnet_best_epoch.pt"))
model.to(device)

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
        outputs_denorm = denormalize_targets(outputs)
        predictions.append(outputs_denorm)
predictions = [item for sublist in predictions for item in sublist]


def save_predictions(predictions, test_csv_dataframe, filename):
    with open(filename, "w") as f:
        f.write("id,X4,X11,X18,X26,X50,X3112\n")
        for pred, id in zip(predictions, test_csv_dataframe["id"]):
            pred = [p.item() for p in pred]
            f.write(f"{id},{','.join([str(p) for p in pred])}\n")


save_predictions(predictions, test_tabular_data, "predictions.csv")
