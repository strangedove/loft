#!/bin/bash
# Loft Setup Script for RunPod
# Usage: curl -sSL https://raw.githubusercontent.com/strangedove/loft/main/scripts/setup-runpod.sh | bash
#
# Environment variables:
#   LOFT_BRANCH - Git branch to checkout (default: main)
#   HF_KEY      - HuggingFace API token for model downloads
#   WANDB_KEY   - Weights & Biases API key for logging

set -e

BRANCH="${LOFT_BRANCH:-main}"

echo "🚀 Setting up Loft on RunPod..."
echo "   Branch: $BRANCH"

# Clone the repo
if [ -d "/workspace/loft" ]; then
    echo "📁 /workspace/loft already exists, pulling latest..."
    cd /workspace/loft
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "📦 Cloning loft repo..."
    git clone -b "$BRANCH" https://github.com/strangedove/loft.git /workspace/loft
    cd /workspace/loft
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📥 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install loft and dependencies
echo "📦 Installing loft and dependencies..."
cd /workspace/loft
uv sync

# Install flash-attn (often needed, compile can take a while)
echo "⚡ Installing flash-attention..."
uv pip install flash-attn --no-build-isolation || echo "⚠️  flash-attn install failed (may need manual install)"

# Authenticate with HuggingFace
if [ -n "$HF_KEY" ]; then
    echo "🤗 Logging into HuggingFace..."
    uv run huggingface-cli login --token "$HF_KEY"
else
    echo "⚠️  HF_KEY not set, skipping HuggingFace login"
fi

# Authenticate with Weights & Biases
if [ -n "$WANDB_KEY" ]; then
    echo "📊 Logging into Weights & Biases..."
    uv run wandb login "$WANDB_KEY"
else
    echo "⚠️  WANDB_KEY not set, skipping W&B login"
fi

# Add to PATH for convenience
echo 'export PATH="/workspace/loft/.venv/bin:$PATH"' >> ~/.bashrc

echo ""
echo "✅ Loft setup complete!"
echo ""
echo "Usage:"
echo "  cd /workspace/loft"
echo "  uv run loft --help"
echo ""
echo "Or activate the venv:"
echo "  source /workspace/loft/.venv/bin/activate"
echo "  loft --help"
