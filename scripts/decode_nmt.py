from tqdm import tqdm

import torch

from seq2seq.transformer.transformer import Transformer
from seq2seq.data.fr_en import tokenizer


def decode(model, src_sentence, max_len=100, device="cpu"):
    model.eval()
    src_tensor = tokenizer.encode(src_sentence).to(device)

    tgt_tokens = [tokenizer.bos_token_id]

    for _ in tqdm(range(max_len)):
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
    num_layers = 6
    num_heads = 8
    embedding_dim = 512
    ffn_hidden_dim = 512
    qk_length = 512
    value_length = 512
    max_length = 200
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
    model_path = "fr_en_euro_latest.pt"
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)["model"]
        )
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        print("Please make sure the model file exists and the path is correct.")
        return

    model.eval()

    # Sentences to translate (from data/nmt/en-fr-small.csv)
    fr_sentences = [
        "Le Parlement européen salue les décisions prises par la Commission européenne, telles que présentées dans ce rapport, y compris celle qui exige, dans un cas précis, le remboursement des sommes allouées et applique donc l'article 88 du traité CECA.",
        "On sait que jusqu'à présent, le Conseil a refusé d'adopter un tel règlement.",
        "Il ne faudrait pas que le nouveau modèle expérimenté à l' heure actuelle par la Commission ait pour conséquence un pur processus de nationalisation, qui annulerait les effets obtenus par notre politique de concurrence.",
        "Si nous voulons qu'une culture juridique existe en Europe, il va sans dire que le droit ne peut être appliqué par la seule Commission, par des organes centraux, mais qu'il doit aussi l'être par les autorités nationales, par les tribunaux nationaux.",
    ]

    en_sentences = [
        "The European Parliament welcomes the decisions taken by the Commission, as set out in this report, including the one which, in a specific case, demands the repayment of the sums allocated and therefore applies Article 88 of the ECSC Treaty.",
        "We know that the Council has so far refused to adopt such a regulation.",
        "The new model currently being tested by the Commission should not result in a pure process of nationalisation which would undo the effects achieved by our competition policy.",
        "If we want a legal culture to exist in Europe, it goes without saying that the law cannot be applied only by the Commission, by central bodies, but must also be applied by the national authorities, by the national courts.",
    ]

    for fr_sentence, en_sentence in zip(fr_sentences, en_sentences):
        translation = decode(model, fr_sentence, max_len=max_length, device=device)
        print(f"French: {fr_sentence}")
        print(f"Ground Truth English: {en_sentence}")
        print(f"Model Translation: {translation}")
        print("-" * 20)


if __name__ == "__main__":
    main()
