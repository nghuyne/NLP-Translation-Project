# plot_loss.py
import torch
import matplotlib.pyplot as plt

def main():
    data = torch.load("loss_history.pt", map_location="cpu")
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]

    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train / Validation Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=200)
    print("Saved loss_curve.png")

if __name__ == "__main__":
    main()
