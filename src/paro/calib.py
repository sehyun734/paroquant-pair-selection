from typing import Iterator

import torch
from torch import Tensor
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from jaxtyping import BFloat16, Float32, Int64


def _load_calib(
    tokenizer: PreTrainedTokenizerBase,
    n_seqs: int,
    seq_len: int,
) -> Int64[Tensor, "n_seqs seq_len"]:
    # Appendix A.2대로 Pile. 문서를 이어붙여 자름
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    n_needed = n_seqs * seq_len

    tok_ids = []
    for doc in dataset:
        tok_ids += tokenizer.encode(doc["text"] + "\n\n", add_special_tokens=False)
        if len(tok_ids) >= n_needed:
            break

    if len(tok_ids) < n_needed:
        raise ValueError(f"토큰 부족: {len(tok_ids)} < {n_needed}")
    return torch.tensor(tok_ids[:n_needed]).view(n_seqs, seq_len)


@torch.no_grad()
def _capture_targets(
    model: PreTrainedModel,
    calib: Int64[Tensor, "n_seqs seq_len"],
    layers: list[int],
    linears: list[str],
) -> list[
    tuple[
        tuple[
            int,  # layer
            str,  # linear
        ],
        tuple[
            Float32[Tensor, "n_toks d_in"],  # x
            Float32[Tensor, "d_out d_in"],  # w
        ],
    ]
]:
    # 대상 linear의 입력과 가중치만 뽑아내면 이후 실험에 모델이 필요 없음
    targets = {
        (i_layer, name): model.get_submodule(f"model.layers.{i_layer}.{name}") for i_layer in layers for name in linears
    }
    captured = {target: [] for target in targets}

    def make_hook(target: tuple[int, str]):
        def hook(
            mod,
            args: tuple[BFloat16[Tensor, "batch seq_len d_in"], ...],
        ) -> None:
            # 모델은 bf16이지만 통계와 최적화는 fp32로. 토큰 축은 전부 펴서 쌓음
            x = args[0]
            captured[target].append(x.reshape(-1, x.shape[-1]).float())

        return hook

    handles = [linear.register_forward_pre_hook(make_hook(target)) for target, linear in targets.items()]

    device = next(model.parameters()).device
    for seq in calib:
        model(seq.unsqueeze(0).to(device))
    for handle in handles:
        handle.remove()

    return [(target, (torch.cat(chunks), targets[target].weight.float())) for target, chunks in captured.items()]


def capture(
    model_name: str,
    layers: list[int],
    linears: list[str],
    n_seqs: int,
    seq_len: int,
) -> list[
    tuple[
        tuple[int, str],
        tuple[
            Float32[Tensor, "n_toks d_in"],
            Float32[Tensor, "d_out d_in"],
        ],
    ]
]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    calib = _load_calib(tokenizer, n_seqs, seq_len)
    captured = _capture_targets(model, calib, layers, linears)

    del model
    torch.cuda.empty_cache()
    return captured


def capture_all(
    models: list[str],
    layers: list[int],
    linears: list[str],
    n_seqs: int,
    seq_len: int,
) -> Iterator[
    tuple[
        str,  # model
        tuple[int, str],
        tuple[
            Float32[Tensor, "n_toks d_in"],
            Float32[Tensor, "d_out d_in"],
        ],
    ]
]:
    for model in models:
        captured = capture(model, layers, linears, n_seqs, seq_len)
        for target, tensors in captured:
            yield model, target, tensors
        del captured
        torch.cuda.empty_cache()
