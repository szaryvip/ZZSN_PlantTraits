NUM_EPOCHS = 20
MAX_VALIDATION_DECREASE_COUNTER = 2
VALIDATE_AFTER_N_BATCHES = 10
SAVE_AFTER_EPOCHS = 5
BATCH_SIZE = 128
PATH_TO_METRICS = {"midfustion": "models/metrics_midfusion.txt",
                   "latefustion": "models/metrics_latefusion.txt",
                   "ensemble": "models/metrics_ensemble.txt"}
COLORS = {"midfustion": "red",
          "latefustion": "blue",
          "ensemble": "green"}
