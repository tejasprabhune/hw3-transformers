import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from seq2seq.transformer.transformer import Decoder
from seq2seq.data.screenplay import ScreenplayDataset, collate_fn, tokenizer

run = wandb.init(
    entity="tejasprabhune-uc-berkeley-electrical-engineering-compute",
    project="transformer",
    config={
        "learning_rate": 0.00005,
        "architecture": "transformer-lm-gpt",
        "dataset": "screenplay",
        "epochs": 10,
    },
)


def decode(model, src_sentence, max_len=100, device="cpu"):
    model.eval()
    tgt_tokens = [tokenizer.bos_token_id]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_tokens]).to(device)
        with torch.no_grad():
            output = model(tgt_tensor)

        next_token_logits = output[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token)

    return tokenizer.decode(torch.tensor(tgt_tokens))


def save_checkpoint(epoch: int, model, optimizer, scheduler, latest = True):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    if latest:
        torch.save(checkpoint, f"screenplay_lm_gpt_latest.pt")
    else:
        torch.save(checkpoint, f"screenplay_lm_gpt_{epoch}.pt")


def make_pad_mask(q, k):
    # k: (B, T_k)
    # returns: (B, 1, 1, T_k)
    pad_mask = k.eq(tokenizer.pad_token_id).unsqueeze(1).unsqueeze(1)
    return pad_mask


def make_no_peak_mask(q, k, device=0):
    # Create a look-ahead mask to prevent attending to future tokens
    len_q, len_k = q.size(1), k.size(1)
    mask = torch.triu(
        torch.ones(len_q, len_k, device=device, dtype=torch.bool), diagonal=1
    )
    return mask


def train_lm():
    data_path = Path("data/lm/")
    dataset = ScreenplayDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    device = 0

    vocab_size = len(tokenizer.vocab)
    num_layers = 6
    num_heads = 8
    embedding_dim = 512
    ffn_hidden_dim = 512
    qk_length = 512
    value_length = 512
    max_length = 1000
    dropout = 0.1
    epochs = 200

    warmup_steps = 4000
    base_lr = 5e-5

    def lr_lambda(step):
        if step == 0:
            step = 1  # avoid div by zero
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return (warmup_steps**0.5) / (step**0.5)

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

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ckpt = torch.load("screenplay_lm_21.pt")
    # model.load_state_dict(ckpt["model"])
    # optimizer.load_state_dict(ckpt["optimizer"])
    # scheduler.load_state_dict(ckpt["scheduler"])

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        data_tqdm = tqdm(dataloader)
        for i, paragraph in enumerate(data_tqdm):
            try:
                paragraph = paragraph.to(device)

                para_input = paragraph[:, :-1]
                para_output = paragraph[:, 1:]

                optimizer.zero_grad()

                trg_pad_mask = make_pad_mask(para_input, para_input)
                trg_no_peak_mask = make_no_peak_mask(para_input, para_input)

                trg_mask = trg_pad_mask | trg_no_peak_mask

                output = model(para_input, tgt_mask=trg_mask)

                loss = criterion(
                    output.reshape(-1, vocab_size), para_output.reshape(-1)
                )
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                data_tqdm.set_postfix({"loss": loss})
                run.log({"loss": loss})
            except Exception as e:
                print(e)

            if i % 5000 == 0:
                print("Saving checkpoint...")
                save_checkpoint(epoch, model, optimizer, scheduler)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Loss - {avg_loss}")

        if epoch % 25 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, latest=False)


if __name__ == "__main__":
    train_lm()
