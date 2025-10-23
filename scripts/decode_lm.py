from tqdm import tqdm

import torch

from seq2seq.transformer.transformer import Decoder
from seq2seq.data.screenplay import tokenizer


def decode(model, start_tokens=None, max_len=200, device="cpu"):
    model.eval()
    if start_tokens is None:
        # Start with the beginning of sequence token if no prompt is given
        tgt_tokens = [tokenizer.bos_token_id]
    else:
        tgt_tokens = start_tokens

    for _ in tqdm(range(max_len)):
        tgt_tensor = torch.tensor([tgt_tokens]).to(device)
        with torch.no_grad():
            output = model(tgt_tensor)

        next_token_logits = output[0, -1, :]
        # Greedy decoding
        next_token = torch.argmax(next_token_logits, dim=-1).item()

        if next_token == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token)

    return tokenizer.decode(torch.tensor(tgt_tokens))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model configuration from train_lm.py
    vocab_size = len(tokenizer.vocab)
    num_layers = 6
    num_heads = 8
    embedding_dim = 512
    ffn_hidden_dim = 512
    qk_length = 512
    value_length = 512
    max_length = 1000
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
    model_path = "screenplay_lm_0.pt"
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
    start_prompt = "LELAND TURBO\n"
    start_tokens = tokenizer.encode(start_prompt).tolist()
    generated_text = decode(
        model, start_tokens=start_tokens, max_len=200, device=device
    )

    # Generate text from scratch
    # generated_text = decode(model, max_len=200, device=device)

    print("\n--- Generated Text ---")
    print(generated_text)
    print("-" * 20)


if __name__ == "__main__":
    main()
