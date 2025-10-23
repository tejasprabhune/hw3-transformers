from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from seq2seq.tokenizer.bpe_tokenizer import BPETokenizer


tokenizer = BPETokenizer(model="gpt2")


class ScreenplayDataset(Dataset):
    def __init__(self, screenplay_path: Path, verbose: bool = False):
        self.paragraphs = []
        for path in screenplay_path.glob("*.txt"):
            num_para = 0
            with open(path, "r") as f:
                current_paragraph = []
                for line in f:
                    if line.isspace():
                        if current_paragraph and current_paragraph[-1][-1].isspace():
                            if num_para == 0:
                                current_paragraph.insert(0, tokenizer.bos_token)
                            self.paragraphs.append("".join(current_paragraph))
                            current_paragraph = []
                            num_para += 1
                        else:
                            current_paragraph.append("\n")
                    else:
                        current_paragraph.append(line)
                self.paragraphs[-1] = self.paragraphs[-1] + tokenizer.eos_token
            if verbose:
                print(path, num_para)

        while "\n" in self.paragraphs:
            self.paragraphs.remove("\n")

        original_len = len(self.paragraphs)

        while len(self.paragraphs) > 0.1 * original_len:
            r_idx = torch.randint(0, len(self.paragraphs) - 1, (1,))[0].item()
            self.paragraphs[r_idx : r_idx + 2] = [
                "".join(self.paragraphs[r_idx : r_idx + 2])
            ]

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx: int):
        para = self.paragraphs[idx]
        # print(repr(para))
        para_tok = tokenizer.encode(para)
        return para_tok


def collate_fn(batch):
    pad_in = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    return pad_in


if __name__ == "__main__":
    data = ScreenplayDataset(Path("data/lm/"), verbose=True)

    idx = torch.randint(0, len(data), (10,)).tolist()

    for i in idx:
        para = data[i]
        print(tokenizer.decode(para))
        print("-" * 80)
