from datasets import load_dataset
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def load_pile(
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
    sequence_len: int,
) -> Tensor:
    # Appendix A.2를 따라 calibration은 pile.
    # 문서를 이어 붙여 sequence_len씩 자름.
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    num_tokens = num_samples * sequence_len

    token_ids = []
    for row in dataset:
        token_ids += tokenizer.encode(row["text"] + "\n\n", add_special_tokens=False)
        if len(token_ids) >= num_tokens:
            break

    if len(token_ids) < num_tokens:
        raise ValueError(f"not enough tokens: {len(token_ids)} < {num_tokens}")
    return torch.tensor(token_ids[:num_tokens]).view(num_samples, sequence_len)
