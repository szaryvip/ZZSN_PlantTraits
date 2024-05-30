# from torcheval.metrics import R2Score
from torchmetrics.regression import R2Score
import torch

from utils import denormalize_targets

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
from dataset import PGLSDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate_model(model, metric, val_data_loader,
                   original_means, original_stds):
    iterations = 0
    accumulated_r2 = 0
    with torch.no_grad():
        for val_data in val_data_loader:
            image, features, targets = val_data
            image = image.to(device)
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(image, features)
            outputs = outputs.to("cpu")
            targets = targets.to("cpu")
            outputs_denorm = denormalize_targets(outputs, original_means,
                                                 original_stds)
            targets_denorm = denormalize_targets(targets, original_means,
                                                 original_stds)
            metric.update(outputs_denorm, targets_denorm)
            accumulated_r2 += metric.compute().item()
            iterations += 1
    return accumulated_r2/iterations


def train_model(model, train_data_loader, val_data_loader, model_path_prefix,
                original_means, original_stds, validation_after_n_batches=20):
    metric = R2Score(num_outputs=6)
    # criterion = torch.nn.functional.mse_loss

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), maximize=False, lr=1e-4,
                                 weight_decay=1e-5)
    model_best_batch_path = f"./models/{model_path_prefix}_best_batch.pt"
    max_value_metrics_epoch = float("-inf")
    model_best_epoch_path = f"./models/{model_path_prefix}_best_epoch.pt"
    model_last_path = f"./models/{model_path_prefix}_last.pt"
    metrics_file = f"./models/metrics_{model_path_prefix}.txt"
    last_value_metrics_validation = float("-inf")
    batch_iteration = 0
    validation_decrease_counter = 0

    model.train()
    for epoch in range(20):
        for data in train_data_loader:
            batch_iteration += 1
            image, features, targets = data
            image = image.to(device)
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(image, features)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            outputs = outputs.to("cpu")
            targets = targets.to("cpu")
            outputs_denorm = denormalize_targets(outputs, original_means,
                                                 original_stds)
            targets_denorm = denormalize_targets(targets, original_means,
                                                 original_stds)
            metric.update(outputs_denorm, targets_denorm)
            new_max_value_metrics_batch = metric.compute().item()
            with open(metrics_file, "a") as f:
                f.write(f"Epoch {epoch}: R2={new_max_value_metrics_batch}\n")

            if batch_iteration % validation_after_n_batches == 0:
                model.eval()
                new_validation_r2 = validate_model(model, metric,
                                                   val_data_loader,
                                                   original_means,
                                                   original_stds)
                metric.reset()
                with open(metrics_file, "a") as f:
                    f.write(f"Validation R2={new_validation_r2}\n")
                if new_validation_r2 < last_value_metrics_validation:
                    validation_decrease_counter += 1
                    if validation_decrease_counter == 15:
                        test_data_loader, original_means, original_stds, tabular_input_size, test_tabular_data = prepare_data_v2()
                        make_predictions(model, test_data_loader, test_tabular_data,
                                         original_means, original_stds)
                        exit(0)
                else:
                    torch.save(model.state_dict(), model_best_batch_path)
                    last_value_metrics_validation = new_validation_r2
                    validation_decrease_counter = 0
                model.train()

        if (new_max_value_metrics_batch > max_value_metrics_epoch):
            max_value_metrics_epoch = new_max_value_metrics_batch
            torch.save(model.state_dict(), model_best_epoch_path)

    torch.save(model.state_dict(), model_last_path)


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


def save_predictions(predictions, test_csv_dataframe, filename):
    with open(filename, "w") as f:
        f.write("id,X4,X11,X18,X26,X50,X3112\n")
        for pred, id in zip(predictions, test_csv_dataframe["id"]):
            pred = [p.item() for p in pred]
            f.write(f"{id},{','.join([str(p) for p in pred])}\n")


def prepare_data_v2():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    batch_size = 128

    test_images_path = "data/test_images"

    train_csv_path = "data/train.csv"
    test_csv_path = "data/test.csv"

    tabular_data = pd.read_csv(train_csv_path)
    targets = ["X4", "X11", "X18", "X26", "X50", "X3112"]

    # Filter data
    upper_values = {}
    for target in targets:
        upper_values[target] = tabular_data[target + "_mean"].quantile(0.95)
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
