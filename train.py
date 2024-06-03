# from torcheval.metrics import R2Score
from torchmetrics.regression import R2Score
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 20
MAX_VALIDATION_DECREASE_COUNTER = 2
VALIDATE_AFTER_N_BATCHES = 10
SAVE_AFTER_EPOCHS = 5


def denormalize_targets(targets, original_means, original_stds):
    return targets * original_stds + original_means


def validate_model(model, metric, val_data_loader,
                   original_means, original_stds):
    iterations = 0
    accumulated_r2 = 0
    with torch.no_grad():
        for val_data in val_data_loader:
            _, image, features, targets = val_data
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
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.functional.mse_loss

    model.to(device)
    try:
        for image_model in model.image_models:
            image_model.to(device)
    except AttributeError:
        pass

    optimizer = torch.optim.Adam(model.parameters(), maximize=False, lr=1e-4,
                                 weight_decay=1e-5)
    model_best_epoch_path = f"./models/{model_path_prefix}_best_epoch.pt"
    model_last_path = f"./models/{model_path_prefix}_last.pt"
    model_best_batch_path = f"./models/{model_path_prefix}_best_batch.pt"
    max_value_metrics_epoch = float("-inf") 
    metrics_file = f"./models/metrics_{model_path_prefix}.txt"
    last_value_metrics_validation = float("-inf")
    validation_decrease_counter = 0
    epoch_counter = 0
    batch_iteration = 0 

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_counter += 1
        for data in train_data_loader:
            batch_iteration += 1 
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

            # if batch_iteration % validation_after_n_batches == 0:
            #     model.eval()
            #     new_validation_r2 = validate_model(model, metric,
            #                                         val_data_loader,
            #                                         original_means,
            #                                         original_stds)
            #     metric.reset()
            #     with open(metrics_file, "a") as f:
            #         f.write(f"Validation R2={new_validation_r2}\n")
            #     if new_validation_r2 < last_value_metrics_validation:
            #         validation_decrease_counter += 1
            #         if validation_decrease_counter == MAX_VALIDATION_DECREASE_COUNTER:
            #             torch.save(model.state_dict(), model_last_path)
            #             exit(0)
            #     else:
            #         torch.save(model.state_dict(), model_best_batch_path)
            #         last_value_metrics_validation = new_validation_r2
            #         validation_decrease_counter = 0
            #     model.train()
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
