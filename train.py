from torcheval.metrics import R2Score
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 20
MAX_VALIDATION_DECREASE_COUNTER = 3


def denormalize_targets(targets, original_means, original_stds):
    return targets * original_stds + original_means


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
    metric = R2Score()
    criterion = torch.nn.MSELoss()

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), maximize=False, lr=1e-4,
                                 weight_decay=1e-5)
    model_best_epoch_path = f"./models/{model_path_prefix}_best_epoch.pt"
    model_last_path = f"./models/{model_path_prefix}_last.pt"
    metrics_file = f"./models/metrics_{model_path_prefix}.txt"
    last_value_metrics_validation = float("-inf")
    validation_decrease_counter = 0

    model.train()
    for epoch in range(NUM_EPOCHS):
        for data in train_data_loader:
            _, image, features, targets = data
            image = image.to(device)
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(image, features)
            loss = criterion(outputs, targets)
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
            if validation_decrease_counter == MAX_VALIDATION_DECREASE_COUNTER:
                torch.save(model.state_dict(), model_last_path)
                exit(0)
        else:
            torch.save(model.state_dict(), model_best_epoch_path)
            last_value_metrics_validation = new_validation_r2
            validation_decrease_counter = 0
        model.train()

    torch.save(model.state_dict(), model_last_path)
