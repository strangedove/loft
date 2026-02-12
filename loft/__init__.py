from typing import TYPE_CHECKING

from .import_utils import _LazyModule


__version__ = '0.1.0'

_import_structure = {
    "scripts": [
        "DataPrepConfig",
        "DatasetMixtureConfig",
        "ScriptArguments",
        "TrlParser",
        "get_dataset",
        "init_zero_verbose",
    ],
    "data_utils": [
        "apply_chat_template",
        "extract_prompt",
        "is_conversational",
        "is_conversational_from_value",
        "maybe_apply_chat_template",
        "maybe_convert_to_chatml",
        "maybe_extract_prompt",
        "maybe_unpair_preference_dataset",
        "pack_dataset",
        "truncate_dataset",
        "unpair_preference_dataset",
    ],
    "models": [
        "clone_chat_template",
        "setup_chat_format",
    ],
    "trainer": [
        "LogCompletionsCallback",
        "ModelConfig",
        "SFTConfig",
        "SFTTrainer",
    ],
    "trainer.callbacks": [
        "MergeModelCallback",
        "RichProgressCallback",
    ],
    "trainer.utils": [
        "get_kbit_device_map",
        "get_peft_config",
        "get_quantization_config",
    ],
}

if TYPE_CHECKING:
    from .data_utils import (
        apply_chat_template,
        extract_prompt,
        is_conversational,
        is_conversational_from_value,
        maybe_apply_chat_template,
        maybe_convert_to_chatml,
        maybe_extract_prompt,
        maybe_unpair_preference_dataset,
        pack_dataset,
        truncate_dataset,
        unpair_preference_dataset,
    )
    from .models import (
        clone_chat_template,
        setup_chat_format,
    )
    from .scripts import (
        DataPrepConfig,
        DatasetMixtureConfig,
        ScriptArguments,
        TrlParser,
        get_dataset,
        init_zero_verbose,
    )
    from .trainer import (
        LogCompletionsCallback,
        ModelConfig,
        SFTConfig,
        SFTTrainer,
    )
    from .trainer.callbacks import (
        MergeModelCallback,
        RichProgressCallback,
    )
    from .trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
