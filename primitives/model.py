import torch
from torch import Tensor
from torch.nn import Module
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_ORGS = {
    "Qwen3-0.6B": "Qwen",
    "Llama-3.1-8B": "meta-llama",
    "Mistral-7B-v0.3": "mistralai",
    "gemma-2-9b": "google",
}


def load_model(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ORGS[model_name]}/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(f"{MODEL_ORGS[model_name]}/{model_name}", dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model, tokenizer


def capture_linears(
    model: PreTrainedModel,
    samples: Tensor,
    layers: list[int],
    linear_names: list[str],
) -> dict[tuple[int, str], tuple[Tensor, Tensor]]:
    # 대상 linear의 입력과 weight만 뽑아 두면 이후에는 모델이 필요 없음.
    linears = {model.get_submodule(f"model.layers.{layer}.{linear_name}"): (layer, linear_name) for layer in layers for linear_name in linear_names}
    inputs = {target: [] for target in linears.values()}

    def hook(
        module: Module,
        args: tuple,
    ) -> None:
        # 모델은 bf16이지만 통계와 최적화는 fp32로 함.
        inputs[linears[module]].append(args[0].reshape(-1, args[0].shape[-1]).float())

    handles = [linear.register_forward_pre_hook(hook) for linear in linears]
    for input_ids in samples:
        model(input_ids.unsqueeze(0).to(model.device), use_cache=False)
    for handle in handles:
        handle.remove()
    return {target: (torch.cat(inputs[target]), linear.weight.float()) for linear, target in linears.items()}
