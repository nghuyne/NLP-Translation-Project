import torch
import torch.nn as nn
import time
from tqdm import tqdm

from .utils import epoch_time, save_checkpoint


def train_one_epoch(model, dataloader, optimizer, criterion, clip, device):

    model.train()
    epoch_loss = 0

    scaler = torch.amp.GradScaler('cuda')

    for src, src_len, trg, trg_len in tqdm(dataloader, desc="Training", leave=False):

        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        # mixed precision giúp giảm 40–60% GPU RAM
        with torch.amp.autocast('cuda'):
            output = model(src, src_len, trg)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    
    torch.cuda.empty_cache()
    return epoch_loss / len(dataloader)



def evaluate(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad(), torch.amp.autocast('cuda'):

        for src, src_len, trg, trg_len in tqdm(dataloader, desc="Validating", leave=False):

            src = src.to(device)
            trg = trg.to(device)

            output = model(src, src_len, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)



def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=10,
    clip=1.0,
    save_path="checkpoints/best_model.pt"
):

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, clip, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)

        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.3f}")
        print(f"  Val   Loss: {val_loss:.3f}")
        print(f"  Time: {mins}m {secs}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, save_path)

    print("\n[✓] Training completed.")
    print(f"[✓] Best validation loss: {best_val_loss:.3f}")
