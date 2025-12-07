import torch
import random
import numpy as np
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pt"):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(state, path)
    print(f"[✓] Saved checkpoint to {path}")

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"[✓] Loaded checkpoint from {path}")
    return checkpoint.get("epoch", 0)

def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)
    return mins, secs

def move_to_device(x, device):
    return x.to(device)
