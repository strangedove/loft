from typing import TYPE_CHECKING

from ..import_utils import _LazyModule


_import_structure = {
    "activation_offloading": ["get_act_offloading_ctx_manager"],
    "utils": [
        "SUPPORTED_ARCHITECTURES",
        "clone_chat_template",
        "prepare_peft_model",
        "setup_chat_format",
    ],
}


if TYPE_CHECKING:
    from .activation_offloading import get_act_offloading_ctx_manager
    from .utils import (
        SUPPORTED_ARCHITECTURES,
        clone_chat_template,
        prepare_peft_model,
        setup_chat_format,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
