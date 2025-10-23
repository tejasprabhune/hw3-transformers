from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from seq2seq.transformer.transformer import Transformer
from seq2seq.data.fr_en import FrEnDataset, collate_fn, tokenizer


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

        if next_token == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token)

    return tokenizer.decode(torch.tensor(tgt_tokens))


def train_nmt():
    data_path = Path("data/nmt/europarl/")
    dataset = FrEnDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    device = 0

    vocab_size = len(tokenizer.vocab)
    num_layers = 6
    num_heads = 8
    embedding_dim = 512
    ffn_hidden_dim = 2048
    qk_length = 512
    value_length = 512
    max_length = 100
    dropout = 0.1
    epochs = 10

    warmup_steps = 4000
    base_lr = 1e-3

    def lr_lambda(step):
        if step == 0:
            step = 1  # avoid div by zero
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return (warmup_steps**0.5) / (step**0.5)

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

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        data_tqdm = tqdm(dataloader)
        for src, tgt in data_tqdm:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            output = model(src, tgt_input)

            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            data_tqdm.set_postfix({"loss": loss})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Loss - {avg_loss}")

    model.eval()
    test_sentences = [
        "où est le restaurant?",
        "je suis un étudiant.",
        "quel temps fait-il aujourd'hui?",
    ]

    for sentence in test_sentences:
        translation = decode(model, sentence, max_len=max_length, device=device)
        print(f"French: {sentence}")
        print(f"English: {translation}")
        print("-" * 20)


if __name__ == "__main__":
    train_nmt()
