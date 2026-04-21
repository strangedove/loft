# ScatterMoE Integration Attribution

This directory contains ScatterMoE kernel code adapted from:

- **axolotl** (https://github.com/axolotl-ai-collective/axolotl)
  - Path: `src/axolotl/integrations/kernels/libs/scattermoe_lora/`
  - License: Apache License 2.0
  - Copyright: axolotl-ai-collective contributors

- **scattermoe** (https://github.com/shawntan/scattermoe)  
  - Original Triton kernels
  - License: Apache License 2.0
  - Copyright: Shawn Tan and ScatterMoE Contributors

## Modifications from upstream axolotl

- Reorganized directory structure (`kernels/` → `triton/`)
- Fixed import paths for loft package structure
- Removed axolotl-specific plugin/config system
- Added `patch.py` integration module for loft's training pipeline
