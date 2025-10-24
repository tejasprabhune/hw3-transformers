from tqdm import tqdm

import torch

from seq2seq.transformer.transformer import Decoder
from seq2seq.data.screenplay import tokenizer
from seq2seq.tokenizer.bpe_tokenizer import BPETokenizer


def decode(model, start_tokens=None, max_len=1000, device="cpu", mode="top_p"):
    model.eval()
    if start_tokens is None:
        # Start with the beginning of sequence token if no prompt is given
        tgt_tokens = [tokenizer.bos_token_id]
    else:
        tgt_tokens = start_tokens

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_tokens]).to(device)
        with torch.no_grad():
            output = model(tgt_tensor)

        next_token_logits = output[0, -1, :]

        if mode == "top_k":
            # top-k
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, 20)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('inf')

            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
        elif mode == "top_p":
            # top-p
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('inf')

            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
        elif mode == "greedy":
            next_token = torch.argmax(next_token_logits).item()

        if next_token == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token)

        print(tokenizer.decode(torch.tensor([next_token])), sep="", end="", flush=True)

    return tokenizer.decode(torch.tensor(tgt_tokens))


def main():
    device = "cuda:3"
    print(f"Using device: {device}")

    # Model configuration from train_lm.py
    vocab_size = len(tokenizer.vocab)
    num_layers = 6
    num_heads = 8
    embedding_dim = 512
    ffn_hidden_dim = 512
    qk_length = 512
    value_length = 512
    max_length = 5000
    dropout = 0.1

    # Instantiate the model
    model = Decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        qk_length=qk_length,
        max_length=max_length,
        value_length=value_length,
        dropout=dropout,
    ).to(device)

    # Load the trained model weights
    model_path = "screenplay_lm_gpt_latest.pt"
    try:
        # The training script saves a checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        print("Please make sure the model file exists and the path is correct.")
        return
    except KeyError:
        # Fallback for models saved directly as state_dict
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading state dict: {e}")
            return

    model.eval()

    # --- Text Generation ---
    print("Generating text from the language model...")

    # Optional: Provide a starting prompt
    start_prompt = """ANAKIN"""
    start_tokens = tokenizer.encode(start_prompt).tolist()
    print(start_prompt, sep="", end="")
    generated_text = decode(
        model, start_tokens=start_tokens, max_len=1000, device=device
    )
    print()

    # Generate text from scratch
    # generated_text = decode(model, max_len=300, device=device)


if __name__ == "__main__":
    main()
