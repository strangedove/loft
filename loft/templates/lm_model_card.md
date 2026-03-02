---
{{ card_data }}
---

# {{ model_name | default("Model", true) }}

{% if base_model and not base_model.startswith("/") and not base_model.startswith("./") and not base_model.startswith("~") -%}
This model is a fine-tuned version of [{{ base_model }}](https://huggingface.co/{{ base_model }}){% if dataset_name %} on the {{ dataset_name }} dataset{% endif %}.
{%- else -%}
This model was fine-tuned{% if dataset_name %} on the {{ dataset_name }} dataset{% endif %} using {{ trainer_name }}.
{%- endif %}

{% if wandb_url -%}
**W&B run:** [{{ wandb_url }}]({{ wandb_url }})
{% endif -%}
{% if comet_url -%}
**Comet experiment:** [{{ comet_url }}]({{ comet_url }})
{% endif -%}

## Training procedure
{% if hyperparams %}
### Hyperparameters

| Parameter | Value |
|-----------|-------|
{% if hyperparams.learning_rate is defined %}| Learning rate | `{{ hyperparams.learning_rate }}` |
{% endif %}{% if hyperparams.lr_scheduler_type is defined %}| LR scheduler | {{ hyperparams.lr_scheduler_type }} |
{% endif %}{% if hyperparams.per_device_train_batch_size is defined %}| Per-device batch size | {{ hyperparams.per_device_train_batch_size }} |
{% endif %}{% if hyperparams.gradient_accumulation_steps is defined and hyperparams.gradient_accumulation_steps > 1 %}| Gradient accumulation | {{ hyperparams.gradient_accumulation_steps }} |
{% endif %}{% if hyperparams.effective_batch_size is defined %}| Effective batch size | {{ hyperparams.effective_batch_size }} |
{% endif %}{% if hyperparams.num_train_epochs is defined %}| Epochs | {{ hyperparams.num_train_epochs }} |
{% endif %}{% if hyperparams.max_length is defined %}| Max sequence length | {{ hyperparams.max_length }} |
{% endif %}{% if hyperparams.optim is defined %}| Optimizer | {{ hyperparams.optim }} |
{% endif %}{% if hyperparams.weight_decay is defined and hyperparams.weight_decay > 0 %}| Weight decay | {{ hyperparams.weight_decay }} |
{% endif %}{% if hyperparams.warmup_ratio is defined and hyperparams.warmup_ratio > 0 %}| Warmup ratio | {{ hyperparams.warmup_ratio }} |
{% endif %}{% if hyperparams.max_grad_norm is defined %}| Max gradient norm | {{ hyperparams.max_grad_norm }} |
{% endif %}{% if hyperparams.bf16 is defined and hyperparams.bf16 %}| Precision | bf16 |
{% elif hyperparams.fp16 is defined and hyperparams.fp16 %}| Precision | fp16 |
{% endif %}{% if hyperparams.gradient_checkpointing is defined and hyperparams.gradient_checkpointing %}| Gradient checkpointing | yes |
{% endif %}{% if hyperparams.loss_type is defined and hyperparams.loss_type != "sigmoid" %}| Loss type | {{ hyperparams.loss_type }} |
{% endif %}{% if hyperparams.packing is defined and hyperparams.packing %}| Packing | yes |
{% endif %}{% if hyperparams.assistant_only_loss is defined and hyperparams.assistant_only_loss %}| Assistant-only loss | yes |
{% endif %}{% if hyperparams.label_smoothing is defined and hyperparams.label_smoothing > 0 %}| Label smoothing | {{ hyperparams.label_smoothing }} |
{% endif %}{% if hyperparams.use_cce is defined and hyperparams.use_cce %}| Chunked cross-entropy | yes |
{% endif %}
{%- endif %}
{% if lora_params %}
### LoRA configuration

| Parameter | Value |
|-----------|-------|
{% if lora_params.r is defined %}| Rank (r) | {{ lora_params.r }} |
{% endif %}{% if lora_params.lora_alpha is defined %}| Alpha | {{ lora_params.lora_alpha }} |
{% endif %}{% if lora_params.lora_dropout is defined and lora_params.lora_dropout > 0 %}| Dropout | {{ lora_params.lora_dropout }} |
{% endif %}{% if lora_params.target_modules is defined %}| Target modules | {{ lora_params.target_modules }} |
{% endif %}{% if lora_params.use_rslora is defined and lora_params.use_rslora %}| rsLoRA | yes |
{% endif %}{% if lora_params.use_dora is defined and lora_params.use_dora %}| DoRA | yes |
{% endif %}{% if lora_params.quantization is defined %}| Quantization | {{ lora_params.quantization }} |
{% endif %}
{%- endif %}
{% if token_stats %}
### Dataset statistics
{% if token_stats.per_dataset %}
| Dataset | Samples | Total tokens | Trainable tokens |
|---------|--------:|-------------:|-----------------:|
{% for name, stats in token_stats.per_dataset.items() %}| {{ name }} | {{ "{:,}".format(stats.samples) }} | {{ "{:,}".format(stats.total_tokens) }} | {{ "{:,}".format(stats.trainable_tokens) }} |
{% endfor %}{% if token_stats.per_dataset | length > 1 %}| **Total** | **{{ "{:,}".format(token_stats.total_samples) }}** | **{{ "{:,}".format(token_stats.total_tokens) }}** | **{{ "{:,}".format(token_stats.trainable_tokens) }}** |
{% endif %}
{%- else %}

- **Samples:** {{ "{:,}".format(token_stats.total_samples) }}
- **Total tokens:** {{ "{:,}".format(token_stats.total_tokens) }}
- **Trainable tokens:** {{ "{:,}".format(token_stats.trainable_tokens) }}
{%- endif %}
{%- endif %}
{% if training_config_yaml %}
<details>
<summary>Training config</summary>

```yaml
{{ training_config_yaml }}
```

</details>
{%- endif %}
{% if data_config_yaml %}
<details>
<summary>Data config</summary>

```yaml
{{ data_config_yaml }}
```

</details>
{%- endif %}

### Framework versions

- Loft: {{ loft_version | default("unknown", true) }}
- Transformers: {{ transformers_version | default("unknown", true) }}
- Pytorch: {{ pytorch_version | default("unknown", true) }}
- Datasets: {{ datasets_version | default("unknown", true) }}
- Tokenizers: {{ tokenizers_version | default("unknown", true) }}
