# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration class for the models.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        model_name_or_path (`str`, *optional*):
            Model checkpoint for weights initialization.
        model_revision (`str`, *optional*, defaults to `"main"`):
            Specific model version to use. It can be a branch name, a tag name, or a commit id.
        dtype (`Literal["auto", "bfloat16", "float16", "float32"]`, *optional*):
            Override the default `torch.dtype` and load the model under this dtype. Possible values are

                - `"bfloat16"`: `torch.bfloat16`
                - `"float16"`: `torch.float16`
                - `"float32"`: `torch.float32`
                - `"auto"`: Automatically derive the dtype from the model's weights.

        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow for custom models defined on the Hub in their own modeling files. This option should only
            be set to `True` for repositories you trust and in which you have read the code, as it will execute code
            present on the Hub on your local machine.
        attn_implementation (`str`, *optional*):
            Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in which case
            you must install this manually by running `pip install flash-attn --no-build-isolation`.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether to use PEFT for training.
        lora_r (`int`, *optional*, defaults to `16`):
            LoRA R value.
        lora_alpha (`int`, *optional*, defaults to `32`):
            LoRA alpha.
        lora_dropout (`float`, *optional*, defaults to `0.05`):
            LoRA dropout.
        lora_target_modules (`Union[str, list[str]]`, *optional*):
            LoRA target modules.
        lora_target_parameters (`Union[str, list[str]]`, *optional*):
            List of target parameters for LoRA.
        lora_modules_to_save (`list[str]`, *optional*):
            Model layers to unfreeze & train.
        lora_task_type (`str`, *optional*, defaults to `"CAUSAL_LM"`):
            Task type to pass for LoRA (use `"SEQ_CLS"` for reward modeling).
        use_rslora (`bool`, *optional*, defaults to `False`):
            Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, instead of
            the original default value of `lora_alpha/r`.
        use_dora (`bool`, *optional*, defaults to `False`):
            Enable [Weight-Decomposed Low-Rank Adaptation (DoRA)](https://huggingface.co/papers/2402.09353). This
            technique decomposes the updates of the weights into two parts, magnitude and direction. Direction is
            handled by normal LoRA, whereas the magnitude is handled by a separate learnable parameter. This can
            improve the performance of LoRA, especially at low ranks. Right now, DoRA only supports linear and Conv2D
            layers. DoRA introduces a bigger overhead than pure LoRA, so it is recommended to merge weights for
            inference.
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            Whether to use 8 bit precision for the base model. Works only with LoRA.
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            Whether to use 4 bit precision for the base model. Works only with LoRA.
        bnb_4bit_quant_type (`str`, *optional*, defaults to `"nf4"`):
            Quantization type (`"fp4"` or `"nf4"`).
        use_bnb_nested_quant (`bool`, *optional*, defaults to `False`):
            Whether to use nested quantization.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use. It can be a branch name, a tag name, or a commit id."},
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_target_parameters: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of target parameters for LoRA."},
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train."},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, "
            "instead of the original default value of `lora_alpha/r`."
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": "Enable Weight-Decomposed Low-Rank Adaptation (DoRA). This technique decomposes the updates of "
            "the weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
            "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
            "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a "
            "bigger overhead than pure LoRA, so it is recommended to merge weights for inference."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8 bit precision for the base model. Works only with LoRA."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4 bit precision for the base model. Works only with LoRA."},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type.", "choices": ["fp4", "nf4"]},
    )
    use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use nested quantization."},
    )
    model_parallel: bool = field(
        default=False,
        metadata={
            "help": "Use device_map='auto' to split the model across all available GPUs. "
            "Useful for quantized models too large for a single GPU without needing "
            "FSDP or DeepSpeed (which are incompatible with QLoRA)."
        },
    )
    max_memory: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Per-device memory budget for model_parallel layer placement, e.g. "
            '{0: "20GiB", 1: "22GiB"}. Passed to accelerate\'s infer_auto_device_map. '
            "If not set and model_parallel is enabled, computed automatically to account "
            "for training overhead (logits, gradients) on the last GPU."
        },
    )
    force_text_only_causal_lm: bool = field(
        default=False,
        metadata={
            "help": "For multimodal-config models (e.g. Gemma 4 31B-it loads as "
            "Gemma4ForConditionalGeneration), force loading the text-only CausalLM "
            "class instead. Skips vision/audio encoder weights entirely (saves "
            "~0.6B params on Gemma 4 31B), uses the simpler Gemma4ForCausalLM forward "
            "path, and avoids the multimodal-output shim. Requires state-dict key "
            "remap from `language_model.model.X` -> `model.X`. When true, "
            "lora_target_modules regex must NOT include `language_model\\.` prefix."
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            "help": "Whether to use low CPU memory when loading the model. Set to False for models with custom "
            "activation functions that have learnable parameters (e.g., xielu) which don't work with meta tensors."
        },
    )
    use_chunked_dpo: bool = field(
        default=False,
        metadata={
            "help": "Use vocab-chunked log-prob computation in DPO to avoid materializing "
            "full [batch, seq, vocab] logits. Reduces peak memory ~30x for large-vocab "
            "models (248K), enabling longer sequences. Trades ~10-20%% speed for memory."
        },
    )
    chunked_dpo_size: int = field(
        default=4096,
        metadata={"help": "Vocab chunk size for chunked DPO log-prob computation."},
    )
    chunked_mlp: bool = field(
        default=False,
        metadata={"help": "Chunk MLP forward pass along sequence dim for memory savings."},
    )
    chunked_mlp_chunks: int = field(
        default=8,
        metadata={"help": "Number of chunks for chunked MLP."},
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Offload activations to CPU during training. Saves ~3-4GB/GPU, ~10%% slower."},
    )
    # --- ScatterMoE experts implementation (MoE models: Gemma4, etc) ---
    #
    # When enabled, loft registers the scattermoe ExpertsInterface and sets
    # ``config._experts_implementation = "scattermoe"`` before model load so
    # experts dispatch through the fused scattermoe kernel. Mirrors
    # ``SFTConfig.use_scattermoe`` so the DPO script can take the same path.
    use_scattermoe: bool = field(
        default=False,
        metadata={
            "help": "Register scattermoe ExpertsInterface and set the config's "
            "experts implementation to scattermoe before model load. Required for "
            "memory-efficient MoE training on Gemma4 etc."
        },
    )
    # --- scattermoe-native LoRA for MoE experts (custom_lora.attach_scattermoe_lora) ---
    #
    # For large MoE models (Gemma4 26B-A4B) PEFT's target_parameters path
    # OOMs because it materializes the full 3D delta weight per forward.
    # ``scattermoe_lora`` attaches a LoRA directly consumed by the scattermoe
    # fused LoRA kernel as plain nn.Parameters on each experts module,
    # bypassing PEFT's ParamWrapper. When enabled in DPO, the reference
    # forward is wrapped with ``disable_scmoe_lora`` so the base-model
    # reference logprobs match the un-adapted experts.
    scattermoe_lora: bool = field(
        default=False,
        metadata={
            "help": "Attach scattermoe-native LoRA to MoE experts (Gemma4 etc). "
            "Required for expert-LoRA training on large MoE models where PEFT's "
            "target_parameters path OOMs. When on, the DPO reference forward also "
            "disables these adapters."
        },
    )
    scmoe_lora_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Rank for scmoe_lora. Defaults to lora_r."},
    )
    scmoe_lora_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "Alpha for scmoe_lora. Defaults to lora_alpha."},
    )
    scmoe_lora_use_rslora: Optional[bool] = field(
        default=None,
        metadata={"help": "rslora for scmoe_lora. Defaults to use_rslora."},
    )
    # Deprecated params
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if self.torch_dtype and not self.dtype:
            warnings.warn(
                "`torch_dtype` is deprecated and will be removed in version 0.27.0, please use `dtype` instead.",
                DeprecationWarning,
            )
            self.dtype = self.torch_dtype

        if hasattr(self.lora_target_modules, "__len__") and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]
