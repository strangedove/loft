import torch
from transformers import BitsAndBytesConfig
from transformers.utils.import_utils import (
    is_peft_available,
    is_rich_available,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from ..model_config import ModelConfig

if is_peft_available():
    from peft import LoraConfig, PeftConfig #pyright: ignore[reportMissingImports, reportUnknownVariableType]

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def peft_module_casting_to_bf16(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


def get_quantization_config(model_args: ModelConfig) -> BitsAndBytesConfig | None:
    if model_args.load_in_4bit:
        # Convert string dtype to torch.dtype (model_args.dtype is a string like "bfloat16")
        compute_dtype = None
        if model_args.dtype is not None and model_args.dtype not in ["auto", None]:
            compute_dtype = getattr(torch, model_args.dtype, torch.bfloat16)
        else:
            compute_dtype = torch.bfloat16  # Default to bfloat16 for 4-bit quantization

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=compute_dtype,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config




def get_peft_config(model_args: ModelConfig) -> "PeftConfig | None":
    if model_args.use_peft is False:
        return None

    if not is_peft_available():
        raise ValueError(
            "You need to have PEFT library installed in your environment, make sure to install `peft`. " + 
            "Make sure to run `pip install -U peft`."
        )

    peft_config = LoraConfig(  # pyright: ignore[reportPossiblyUnboundVariable]
        task_type=model_args.lora_task_type,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_rslora=model_args.use_rslora,
        use_dora=model_args.use_dora,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config