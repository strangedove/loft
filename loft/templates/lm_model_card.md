---
{{ card_data }}
---

# {{ model_name | default("Model", true) }}

This model is a fine-tuned version of [{{ base_model }}](https://huggingface.co/{{ base_model }}){% if dataset_name %} on the {{ dataset_name }} dataset{% endif %}.

## Training procedure

### Framework versions

- Loft: {{ loft_version | default("unknown", true) }}
- Transformers: {{ transformers_version | default("unknown", true) }}
- Pytorch: {{ pytorch_version | default("unknown", true) }}
- Datasets: {{ datasets_version | default("unknown", true) }}
- Tokenizers: {{ tokenizers_version | default("unknown", true) }}
