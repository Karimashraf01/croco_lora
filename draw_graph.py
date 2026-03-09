import json
import os
import matplotlib.pyplot as plt

root_dir = "output"

for exp in os.listdir(root_dir):

    exp_path = os.path.join(root_dir, exp)

    if not os.path.isdir(exp_path):
        continue

    train_file = None
    val_file = None

    # Find train/val logs
    for f in os.listdir(exp_path):
        if "log_train" in f and f.endswith(".txt"):
            train_file = os.path.join(exp_path, f)

        if "log_val" in f and f.endswith(".txt"):
            val_file = os.path.join(exp_path, f)

    if train_file is None or val_file is None:
        print(f"Skipping {exp} (logs not found)")
        continue

    train_epochs = []
    train_loss = []

    val_epochs = []
    val_loss = []

    # Read training log
    with open(train_file) as f:
        for line in f:
            data = json.loads(line)
            if "train_loss" in data:
                train_epochs.append(data["epoch"])
                train_loss.append(data["train_loss"])

    # Read validation log
    with open(val_file) as f:
        for line in f:
            data = json.loads(line)
            if "val_val_loss" in data:
                val_epochs.append(data["epoch"])
                val_loss.append(data["val_val_loss"])

    if len(train_loss) == 0 or len(val_loss) == 0:
        print(f"Skipping {exp} (no data)")
        continue

    best_val = min(val_loss)
    best_epoch = val_epochs[val_loss.index(best_val)]

    plt.figure(figsize=(10,6))

    plt.plot(train_epochs, train_loss, marker="o", label="Training Loss")
    plt.plot(val_epochs, val_loss, marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(exp)

    plt.legend()
    plt.grid(True)

    # Zoom scale
    all_losses = train_loss + val_loss
    plt.ylim(min(all_losses)-0.005, max(all_losses)+0.005)

    # Text under graph
    plt.text(
        0.5,
        -0.15,
        f"Best Validation Loss: {best_val:.6f} (Epoch {best_epoch})",
        transform=plt.gca().transAxes,
        ha="center",
        fontsize=12
    )

    plt.tight_layout()

    save_path = os.path.join(exp_path, f"loss_curve_{exp}.png")
    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Saved graph for {exp}")