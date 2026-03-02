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

import json
import logging
import os
from typing import Optional, Union

import yaml
from transformers import Trainer, is_wandb_available

from .utils import generate_model_card, get_comet_experiment_url, _is_local_path, _sanitize_config_dict


if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)


class BaseTrainer(Trainer):
    _tag_names = []
    _name = "Base"
    _paper = {}
    _template_file = None

    # ── helpers for model card data ──────────────────────────────────

    def _build_hyperparams(self) -> dict:
        """Extract training hyperparameters from self.args for the model card."""
        a = self.args
        hp = {}
        hp["learning_rate"] = a.learning_rate
        hp["lr_scheduler_type"] = str(a.lr_scheduler_type)
        hp["per_device_train_batch_size"] = a.per_device_train_batch_size
        hp["gradient_accumulation_steps"] = a.gradient_accumulation_steps
        hp["effective_batch_size"] = (
            a.per_device_train_batch_size
            * a.gradient_accumulation_steps
            * max(1, int(os.environ.get("WORLD_SIZE", "1")))
        )
        hp["num_train_epochs"] = a.num_train_epochs
        if hasattr(a, "max_length") and a.max_length is not None:
            hp["max_length"] = a.max_length
        hp["optim"] = str(a.optim)
        hp["weight_decay"] = a.weight_decay
        hp["warmup_ratio"] = a.warmup_ratio
        hp["max_grad_norm"] = a.max_grad_norm
        hp["bf16"] = getattr(a, "bf16", False)
        hp["fp16"] = getattr(a, "fp16", False)
        hp["gradient_checkpointing"] = getattr(a, "gradient_checkpointing", False)
        # SFT-specific fields
        if hasattr(a, "loss_type"):
            hp["loss_type"] = a.loss_type
        if hasattr(a, "packing"):
            hp["packing"] = a.packing
        if hasattr(a, "assistant_only_loss"):
            hp["assistant_only_loss"] = a.assistant_only_loss
        if hasattr(a, "label_smoothing"):
            hp["label_smoothing"] = a.label_smoothing
        if hasattr(a, "use_cce"):
            hp["use_cce"] = a.use_cce
        return hp

    def _build_lora_params(self) -> Optional[dict]:
        """Extract LoRA config from the model if it's a PeftModel."""
        try:
            from peft import PeftModel
        except ImportError:
            return None
        if not isinstance(self.model, PeftModel):
            return None

        peft_config = self.model.peft_config.get("default")
        if peft_config is None:
            return None

        lp = {}
        lp["r"] = getattr(peft_config, "r", None)
        lp["lora_alpha"] = getattr(peft_config, "lora_alpha", None)
        lp["lora_dropout"] = getattr(peft_config, "lora_dropout", 0)
        target = getattr(peft_config, "target_modules", None)
        if target is not None:
            lp["target_modules"] = ", ".join(sorted(target)) if isinstance(target, (set, list)) else str(target)
        lp["use_rslora"] = getattr(peft_config, "use_rslora", False)
        lp["use_dora"] = getattr(peft_config, "use_dora", False)

        # Quantization info
        base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        quant_config = getattr(getattr(base_model, "config", None), "quantization_config", None)
        if quant_config is not None:
            quant_dict = quant_config.to_dict() if hasattr(quant_config, "to_dict") else {}
            if quant_dict.get("load_in_4bit"):
                lp["quantization"] = f"4-bit ({quant_dict.get('bnb_4bit_quant_type', 'nf4')})"
            elif quant_dict.get("load_in_8bit"):
                lp["quantization"] = "8-bit"
        return lp

    def _load_token_stats(self) -> Optional[dict]:
        """Load token statistics from blend_metadata.json if available.

        Normalizes the stored format (train_total_tokens, etc.) to what the
        model card template expects (total_tokens, trainable_tokens, total_samples).
        """
        prepared = getattr(self.args, "prepared_dataset", None)
        if not prepared:
            return None
        meta_path = os.path.join(prepared, "blend_metadata.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
            raw = metadata.get("token_stats", {})
            if not raw:
                return None
            # Normalize keys for the template
            result = {
                "total_tokens": raw.get("train_total_tokens", 0),
                "trainable_tokens": raw.get("train_trainable_tokens", 0),
                "total_samples": metadata.get("splits", {}).get("train", 0),
            }
            if "per_dataset" in raw:
                result["per_dataset"] = raw["per_dataset"]
            return result
        except Exception:
            return None

    def _build_training_config_yaml(self) -> Optional[str]:
        """Read and sanitize the original training config YAML."""
        config_path = getattr(self.args, "_config_path", None)
        if not config_path or not os.path.exists(config_path):
            return None
        try:
            from loft.scripts.utils import resolve_config_inheritance
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            config = resolve_config_inheritance(config, config_path)
            config = _sanitize_config_dict(config)
            return yaml.dump(config, default_flow_style=False, sort_keys=False).strip()
        except Exception:
            logger.debug("Failed to load training config for model card", exc_info=True)
            return None

    def _build_data_config_yaml(self) -> Optional[str]:
        """Read and sanitize the data config YAML."""
        data_config_path = getattr(self.args, "data_config", None)
        if not data_config_path or not os.path.exists(data_config_path):
            return None
        try:
            with open(data_config_path) as f:
                config = yaml.safe_load(f) or {}
            config = _sanitize_config_dict(config)
            return yaml.dump(config, default_flow_style=False, sort_keys=False).strip()
        except Exception:
            logger.debug("Failed to load data config for model card", exc_info=True)
            return None

    # ── model card creation ──────────────────────────────────────────

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Optional[Union[str, list[str]]] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*):
                Name of the model.
            dataset_name (`str`, *optional*):
                Name of the dataset used for training.
            tags (`str`, `list[str]`, *optional*):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # Normalize tags
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)
        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")
        if "JOB_ID" in os.environ:
            tags.add("hf_jobs")
        tags.update(self._tag_names)
        tags = list(tags)

        # Gather extended model card data
        hyperparams = self._build_hyperparams()
        lora_params = self._build_lora_params()
        token_stats = self._load_token_stats()
        training_config_yaml = self._build_training_config_yaml()
        data_config_yaml = self._build_data_config_yaml()

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name=self._name,
            trainer_citation=self._paper.get("citation"),
            template_file=self._template_file,
            paper_title=self._paper.get("title"),
            paper_id=self._paper.get("id"),
            hyperparams=hyperparams,
            lora_params=lora_params,
            token_stats=token_stats,
            training_config_yaml=training_config_yaml,
            data_config_yaml=data_config_yaml,
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))
