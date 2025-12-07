import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

def display_attention(sentence, translation, attention, save_path=None):
    """
    sentence: str (source)
    translation: str (target)
    attention: torch.Tensor (trg_len, src_len)
    """
    
    if attention is None:
        print("[!] No attention weights available (Beam search not supported for viz yet).")
        return

    # Move to CPU and numpy
    attention = attention.cpu().detach().numpy()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)
    
    # Set up axes
    # ax.set_xticklabels([''] + sentence.split(' ') + ['<eos>'], rotation=90)
    # ax.set_yticklabels([''] + translation.split(' ') + ['<eos>'])
    
    # Use FixedLocator to avoid UserWarning
    x_labels = [''] + sentence.split(' ') + ['<eos>']
    y_labels = [''] + translation.split(' ') + ['<eos>']
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(x_labels))))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_labels))
    
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(len(y_labels))))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(y_labels))
    
    plt.show()
    
    if save_path:
        plt.savefig(save_path)
        print(f"[✓] Attention map saved to {save_path}")
