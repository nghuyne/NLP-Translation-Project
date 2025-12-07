# src/train.py
import torch
import torch.nn as nn


def train_epoch(model, loader, optimizer, criterion, clip, device, teacher_forcing):
    model.train()
    total_loss = 0.0

    for i, (src, src_lens, trg, _) in enumerate(loader, start=1):
        src, src_lens, trg = src.to(device), src_lens.to(device), trg.to(device)

        optimizer.zero_grad()
        outputs = model(src, src_lens, trg, teacher_forcing_ratio=teacher_forcing)
        # outputs: [batch, trg_len, vocab]
        output_dim = outputs.shape[-1]

        outputs = outputs[:, 1:, :].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(outputs, trg_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

        # in log mỗi 100 batch để biết đang train
        if i % 100 == 0:
            print(f"   [train] batch {i}/{len(loader)} loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for src, src_lens, trg, _ in loader:
        src, src_lens, trg = src.to(device), src_lens.to(device), trg.to(device)

        outputs = model(src, src_lens, trg, teacher_forcing_ratio=0.0)
        output_dim = outputs.shape[-1]

        outputs = outputs[:, 1:, :].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(outputs, trg_flat)
        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    n_epochs,
    lr,
    clip,
    teacher_forcing,
    trg_pad_idx,
    device,
    save_path="best_model.pth",
    patience=3,
):
    """
    patience: số epoch liên tiếp val_loss không giảm thì dừng (early stopping)
    """
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    no_improve = 0

    train_history = []
    val_history = []

    for epoch in range(1, n_epochs + 1):
        print(f"\n=== Epoch {epoch}/{n_epochs} ===")
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, clip, device, teacher_forcing
        )
        val_loss = eval_epoch(model, val_loader, criterion, device)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved {save_path}")
        else:
            no_improve += 1
            print(f"  (no improvement for {no_improve} epoch(s))")
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # lưu history để vẽ biểu đồ loss
    torch.save(
        {"train_losses": train_history, "val_losses": val_history},
        "loss_history.pt",
    )
    print("Saved loss_history.pt")
