import torch

from seq2seq.transformer.transformer import Transformer
from seq2seq.data.fr_en import tokenizer


def decode(model, src_sentence, max_len=100, device="cpu"):
    model.eval()
    src_tensor = tokenizer.encode(src_sentence).to(device)

    tgt_tokens = [tokenizer.bos_token_id]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_tokens]).to(device)
        with torch.no_grad():
            output = model(src_tensor.unsqueeze(0), tgt_tensor)

        next_token_logits = output[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1).item()
        next_token = torch.argmax(next_token_probs).item()

        if next_token == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token)

    return tokenizer.decode(torch.tensor(tgt_tokens))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model configuration
    vocab_size = len(tokenizer.vocab)
    num_layers = 2
    num_heads = 2
    embedding_dim = 64
    ffn_hidden_dim = 64
    qk_length = 64
    value_length = 64
    max_length = 1500
    dropout = 0.1

    # Instantiate the model
    model = Transformer(
        pad_idx=tokenizer.pad_token_id,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        qk_length=qk_length,
        max_length=max_length,
        value_length=value_length,
        dropout=dropout,
        device=device,
    ).to(device)

    # Load the trained model weights
    model_path = "fr_en_small_e1.pt"
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        print("Please make sure the model file exists and the path is correct.")
        return

    model.eval()

    # Sentences to translate (from data/nmt/en-fr-small.csv)
    test_sentences = [
        ("Plan du site", "Site map"),
        ("Rétroaction", "Feedback"),
        ("Crédits", "Credits"),
        ("Français", "English"),
        (
            "Pour le public, le travail n'est pas sans intérêt : on croit en effet que la position des objets célestes a un impact sur les évènements qui ont cours sur la Terre.",
            "For the public, the work is not without interest: it is believed that the position of celestial objects has an impact on events taking place on Earth.",
        ),
    ]

    for fr_sentence, en_ground_truth in test_sentences:
        translation = decode(model, fr_sentence, max_len=max_length, device=device)
        print(f"French: {fr_sentence}")
        print(f"Ground Truth English: {en_ground_truth}")
        print(f"Model Translation: {translation}")
        print("-" * 20)


if __name__ == "__main__":
    main()
