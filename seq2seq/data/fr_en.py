from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd

from seq2seq.tokenizer.bpe_tokenizer import BPETokenizer


tokenizer = BPETokenizer()


class FrEnDataset(Dataset):
    def __init__(self, fr_en_path: Path):
        self.fr_en_csv = pd.read_csv(fr_en_path)
        self.fr_en_csv = self.fr_en_csv.dropna()

    def __len__(self):
        return len(self.fr_en_csv)

    def __getitem__(self, idx: int):
        row = self.fr_en_csv.iloc[idx]

        fr = row["fr"]
        en = row["en"]

        fr_tok = tokenizer.encode(fr)
        en_tok = tokenizer.encode(en)

        return torch.cat(
            [
                torch.tensor([tokenizer.bos_token_id]),
                fr_tok,
                torch.tensor([tokenizer.eos_token_id]),
            ]
        ), torch.cat(
            [
                torch.tensor([tokenizer.bos_token_id]),
                en_tok,
                torch.tensor([tokenizer.eos_token_id]),
            ]
        )


def collate_fn(batch):
    in_seq = [item[0] for item in batch]
    target_seq = [item[1] for item in batch]

    pad_in = pad_sequence(
        in_seq, batch_first=True, padding_value=tokenizer.eos_token_id
    )
    pad_target = pad_sequence(
        target_seq, batch_first=True, padding_value=tokenizer.eos_token_id
    )

    return pad_in, pad_target
