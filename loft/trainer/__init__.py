from typing import TYPE_CHECKING

from ..import_utils import _LazyModule


_import_structure = {
    "callbacks": [
        "LogCompletionsCallback",
        "MergeModelCallback",
        "RichProgressCallback",
    ],
    "model_config": ["ModelConfig"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "utils": [
        "RunningMoments",
        "compute_accuracy",
        "disable_dropout_in_model",
        "empty_cache",
        "peft_module_casting_to_bf16",
    ],
}

if TYPE_CHECKING:
    from .callbacks import (
        LogCompletionsCallback,
        MergeModelCallback,
        RichProgressCallback,
    )
    from .model_config import ModelConfig
    from .sft_config import SFTConfig
    from .sft_trainer import SFTTrainer
    from .utils import (
        RunningMoments,
        compute_accuracy,
        disable_dropout_in_model,
        empty_cache,
        peft_module_casting_to_bf16,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
