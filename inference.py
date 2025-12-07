import torch
from src.dataset import get_tokenizers
from src.translate import translate_sentence as raw_translate
from src.model import Encoder, Decoder, Seq2Seq
from src.utils import EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, MAX_VOCAB, MIN_FREQ
from src.vocab import PAD

# Global variables to hold model and resources
_model = None
_tok_en = None
_tok_fr = None
_src_stoi = None
_trg_stoi = None
_trg_itos = None
_device = None

def load_resources():
    """
    Load model and vocabularies only once.
    """
    global _model, _tok_en, _tok_fr, _src_stoi, _trg_stoi, _trg_itos, _device
    
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocab
    try:
        vocab_obj = torch.load("vocab.pt", map_location=_device)
        _src_stoi = vocab_obj["src_stoi"]
        src_itos = vocab_obj["src_itos"]
        _trg_stoi = vocab_obj["trg_stoi"]
        _trg_itos = vocab_obj["trg_itos"]
    except FileNotFoundError:
        print("Error: vocab.pt not found. Please train the model first.")
        return

    src_pad_idx = _src_stoi[PAD]
    trg_pad_idx = _trg_stoi[PAD]

    # Initialize Tokenizers
    _tok_en, _tok_fr = get_tokenizers()

    # Initialize Model
    encoder = Encoder(len(src_itos), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, src_pad_idx)
    decoder = Decoder(len(_trg_itos), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, trg_pad_idx)
    _model = Seq2Seq(encoder, decoder, _device, trg_pad_idx).to(_device)

    # Load Checkpoint
    try:
        state = torch.load("best_model.pth", map_location=_device)
        _model.load_state_dict(state)
        _model.eval()
    except FileNotFoundError:
        print("Error: best_model.pth not found. Please train the model first.")

def translate(sentence: str) -> str:
    """
    Yêu cầu bắt buộc: Hàm nhận câu tiếng Anh, trả về câu tiếng Pháp.
    """
    if _model is None:
        load_resources()
    
    if _model is None:
        return "Error: Model could not be loaded."

    result = raw_translate(
        sentence, 
        _model, 
        _tok_en, 
        _tok_fr, 
        _src_stoi, 
        _trg_stoi, 
        _trg_itos, 
        _device
    )
    return result

if __name__ == "__main__":
    # Test nhanh
    sent = "a man is walking"
    print(f"Input: {sent}")
    print(f"Output: {translate(sent)}")
