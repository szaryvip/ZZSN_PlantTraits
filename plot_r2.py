import matplotlib.pyplot as plt


PATH_TO_METRICS = {"midfustion": "models/metrics_midfusion.txt",
                   "latefustion": "models/metrics_latefusion.txt",
                   "ensemble": "models/metrics_ensemble.txt"}
COLORS = {"midfustion": "red",
          "latefustion": "blue",
          "ensemble": "green"}


val_for_model = {}

for model_type in PATH_TO_METRICS.keys():
    train_r2 = []
    val_r2 = []
    with open(PATH_TO_METRICS[model_type], "r") as metrics_file:
        current_epoch = 0
        accumulated_r2 = 0
        iterations = 0
        for metrics_line in metrics_file.readlines():
            if "Epoch" in metrics_line:
                epoch, score = metrics_line.split(":")
                epoch = int(epoch.split(" ")[-1])
                score = score.split("=")[-1]
                score = float(score)
                if epoch == current_epoch:
                    iterations += 1
                    accumulated_r2 += score
                else:
                    train_r2.append(accumulated_r2/iterations)
                    current_epoch = epoch
                    iterations = 1
                    accumulated_r2 = score
            else:
                score = metrics_line.split("=")[-1]
                score = float(score)
                val_r2.append(score)
        train_r2.append(accumulated_r2/iterations)

    x = list(range(len(train_r2)))
    plt.figure()
    plt.plot(x, train_r2, marker="o", color="blue", label="Train R2")
    plt.plot(x, val_r2, marker="x", color="red", label="Validation R2")
    plt.title(f"R2 Score during training - {model_type}")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.xticks(x)
    plt.legend()
    plt.savefig(f"{model_type}_plot.png")

    val_for_model[model_type] = val_r2
    
plt.figure()
for model_type in val_for_model.keys():
    x = list(range(len(val_for_model[model_type])))
    plt.plot(x, val_for_model[model_type], color=COLORS[model_type], label=model_type)

plt.title("Validation R2 for each model")
plt.xlabel("Epochs")
plt.xticks(list(range(20)))
plt.ylabel("R2 Score")
plt.legend()
plt.savefig("combined_plot.png")
