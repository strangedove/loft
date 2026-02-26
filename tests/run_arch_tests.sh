#!/usr/bin/env bash
# Architecture & feature compatibility tests for loft on Transformers 5.x
#
# Usage:
#   bash tests/run_arch_tests.sh           # run all tests
#   bash tests/run_arch_tests.sh 1 3 5     # run specific test numbers
#
# Results are written to tests/arch_results.txt
# Each test runs max_steps=3 with a tiny synthetic dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="$SCRIPT_DIR/arch_results.txt"
WORK_DIR=$(mktemp -d /tmp/loft-arch-XXXXXX)
TIMEOUT_SECONDS=600  # 10 min per test (larger models need more time)
PYTHON="/home/aibox/.venv/bin/python"
LOFT="/home/aibox/.venv/bin/loft"
export PATH="/home/aibox/.venv/bin:$PATH"

# Small default model for feature tests
QWEN05B="/home/aibox/models/Qwen2-0.5B"

# HF cache models (resolved at runtime)
HF_CACHE="$HOME/.cache/huggingface/hub"

resolve_model() {
    local hf_name="$1"
    local dir_name="models--$(echo "$hf_name" | sed 's|/|--|g')"
    local snap_dir="$HF_CACHE/$dir_name/snapshots"
    if [ -d "$snap_dir" ]; then
        # Get the latest snapshot (find dirs, most recently modified first)
        local found
        found=$(ls -1td "$snap_dir"/* 2>/dev/null | head -1)
        if [ -n "$found" ] && [ -d "$found" ]; then
            echo "$found"
        else
            echo ""
        fi
    else
        echo ""
    fi
}

# Parse test numbers from args, or run all
if [ $# -gt 0 ]; then
    TESTS_TO_RUN=("$@")
else
    TESTS_TO_RUN=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
fi

echo "================================================================"
echo "Loft Architecture & Feature Tests"
echo "================================================================"
echo "Work dir: $WORK_DIR"
echo "Results:  $RESULTS_FILE"
echo "Tests:    ${TESTS_TO_RUN[*]}"
echo "================================================================"
echo ""

# Write header to results file
{
    echo "================================================================"
    echo "Loft Architecture & Feature Test Results"
    echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "================================================================"
    echo ""
} > "$RESULTS_FILE"

# ---- Create synthetic dataset ----
DATASET_DIR="$WORK_DIR/dataset"
mkdir -p "$DATASET_DIR"
$PYTHON - "$DATASET_DIR" <<'PYSCRIPT'
import json, sys, os
out = sys.argv[1]
samples = []
for i in range(100):
    samples.append({
        "messages": [
            {"role": "system", "content": "You are a helpful math assistant that shows your work step by step."},
            {"role": "user", "content": f"Please calculate {i * 7} + {i * 3} and explain your reasoning in detail."},
            {"role": "assistant", "content": f"Let me work through this step by step.\n\nFirst, I need to add {i * 7} and {i * 3}.\n\nBreaking it down:\n- {i * 7} + {i * 3} = {i * 7 + i * 3}\n\nSo the answer is {i * 10}. This is because {i} times 7 is {i * 7} and {i} times 3 is {i * 3}, and when you add them together you get {i} times 10 which equals {i * 10}."}
        ]
    })
with open(os.path.join(out, "train.jsonl"), "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")
print(f"Created {len(samples)} synthetic samples at {out}/train.jsonl")
PYSCRIPT

# ---- Helper: write data.yaml ----
write_data_config() {
    local dir="$1"
    cat > "$dir/data.yaml" <<YAML
datasets:
  - path: $DATASET_DIR/train.jsonl
    split: train
shuffle_datasets: false
shuffle_combined: false
eval_split: 0.0
assistant_only_loss: true
YAML
}

# ---- Helper: write training config ----
write_train_config() {
    local dir="$1"
    local model="$2"
    local extra_yaml="${3:-}"
    cat > "$dir/train.yaml" <<YAML
model_name_or_path: $model
data_config: $dir/data.yaml
prepared_dataset: $dir/prepared

bf16: true
gradient_checkpointing: true
max_length: 256

per_device_train_batch_size: 1
gradient_accumulation_steps: 1
max_steps: 3
learning_rate: 2.0e-5
warmup_ratio: 0.0
lr_scheduler_type: constant
logging_steps: 1
save_strategy: "no"
report_to: "none"
output_dir: $dir/output

use_peft: true
lora_r: 8
lora_alpha: 16
lora_dropout: 0.0
load_in_4bit: false
$extra_yaml
YAML
}

# ---- Helper: write FSDP accelerate config ----
write_fsdp_config() {
    local dir="$1"
    cat > "$dir/fsdp.yaml" <<YAML
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
machine_rank: 0
main_process_port: 29500
num_machines: 1
num_processes: 2
use_cpu: false
YAML
}

# ---- Run a single test ----
run_test() {
    local test_num="$1"
    local test_name="$2"
    local description="$3"
    local cmd="$4"

    echo "------------------------------------------------------------"
    echo "TEST $test_num: $test_name"
    echo "  $description"
    echo "------------------------------------------------------------"

    echo "------------------------------------------------------------" >> "$RESULTS_FILE"
    echo "TEST $test_num: $test_name" >> "$RESULTS_FILE"
    echo "Description: $description" >> "$RESULTS_FILE"
    echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$RESULTS_FILE"

    local log_file="$WORK_DIR/test_${test_num}.log"
    local exit_code=0
    timeout $TIMEOUT_SECONDS bash -c "$cmd" > "$log_file" 2>&1 || exit_code=$?

    local status=""
    if [ $exit_code -eq 0 ]; then
        status="PASS"
        echo "  => PASS"
    elif [ $exit_code -eq 124 ]; then
        status="TIMEOUT"
        echo "  => TIMEOUT (exceeded ${TIMEOUT_SECONDS}s)"
    else
        status="FAIL (exit code $exit_code)"
        echo "  => FAIL (exit code $exit_code)"
    fi

    {
        echo "Status: $status"
        echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo ""
        echo "--- Last 30 lines of log ---"
        tail -30 "$log_file" 2>/dev/null || echo "(no log output)"
        echo ""
        echo ""
    } >> "$RESULTS_FILE"

    $PYTHON -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 2
    return 0
}

# ====================================================================
# TEST DEFINITIONS
# ====================================================================

for test_num in "${TESTS_TO_RUN[@]}"; do
    case "$test_num" in

        # ---- Architecture Tests ----

        1)  # Qwen3 0.6B — new Qwen3 architecture
            MODEL=$(resolve_model "Qwen/Qwen3-0.6B")
            if [ -z "$MODEL" ]; then echo "SKIP test 1: Qwen3-0.6B not in cache"; continue; fi
            DIR="$WORK_DIR/t1"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$MODEL"
            run_test 1 "qwen3-0.6b-lora" \
                "Qwen3 0.6B, LoRA, single GPU" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        2)  # Qwen3-VL-2B — multimodal, train text part only
            MODEL=$(resolve_model "Qwen/Qwen3-VL-2B-Instruct")
            if [ -z "$MODEL" ]; then echo "SKIP test 2: Qwen3-VL-2B not in cache"; continue; fi
            DIR="$WORK_DIR/t2"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$MODEL"
            run_test 2 "qwen3-vl-2b-lora" \
                "Qwen3-VL 2B (multimodal), LoRA on text, single GPU" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        3)  # Voxtral-Mini-3B (Mistral text-only) — tests mistral architecture
            MODEL=$(resolve_model "minpeter/Voxtral-Mini-3B-Text-2507-hf")
            if [ -z "$MODEL" ]; then echo "SKIP test 3: Voxtral-Mini-3B not in cache"; continue; fi
            DIR="$WORK_DIR/t3"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$MODEL"
            run_test 3 "voxtral-3b-lora" \
                "Voxtral-Mini 3B (Mistral arch), LoRA, single GPU" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        4)  # Trinity-Nano-Preview-SCM — ScatterMoE (AfmoeSCM), MoE aux losses
            MODEL=$(resolve_model "IntervitensInc/Trinity-Nano-Preview-SCM")
            if [ -z "$MODEL" ]; then echo "SKIP test 4: Trinity-Nano-SCM not in cache"; continue; fi
            DIR="$WORK_DIR/t4"; mkdir -p "$DIR"
            write_data_config "$DIR"
            # Trust remote code for custom architecture. LoRA (not QLoRA) — scattermoe
            # backward pass has bf16/fp32 dtype mismatch with 4-bit quantization.
            write_train_config "$DIR" "$MODEL" "trust_remote_code: true"
            run_test 4 "trinity-nano-scm-lora" \
                "Trinity-Nano-SCM (MoE), LoRA, single GPU, trust_remote_code" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        5)  # Ministral3 14B — multimodal (Mistral3ForConditionalGeneration), text train
            # Use the text-only converted version if available, otherwise the raw multimodal
            MODEL=$(resolve_model "estrogen/SomehowMinistralReturnedButCommunist")
            if [ -z "$MODEL" ]; then
                MODEL=$(resolve_model "mistralai/Ministral-3-14B-Instruct-2512-BF16")
            fi
            if [ -z "$MODEL" ]; then echo "SKIP test 5: No Ministral3 model in cache"; continue; fi
            DIR="$WORK_DIR/t5"; mkdir -p "$DIR"
            write_data_config "$DIR"
            # QLoRA to fit on 2x3090
            write_train_config "$DIR" "$MODEL" "load_in_4bit: true"
            run_test 5 "ministral3-14b-qlora" \
                "Ministral3 14B, QLoRA, single GPU" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        6)  # Devstral 24B — multimodal (Mistral3ForConditionalGeneration), large model
            # This model is FP8-quantized natively. On non-FP8 hardware (< sm_89),
            # loft clears the FP8 config and uses QLoRA 4-bit instead.
            MODEL=$(resolve_model "mistralai/Devstral-Small-2-24B-Instruct-2512")
            if [ -z "$MODEL" ]; then echo "SKIP test 6: Devstral-24B not in cache"; continue; fi
            DIR="$WORK_DIR/t6"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$MODEL" "load_in_4bit: true
model_parallel: true"
            run_test 6 "devstral-24b-qlora-mp" \
                "Devstral 24B, QLoRA, model_parallel across 2 GPUs" \
                "$PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        7)  # Moonlight 16B-A3B — DeepSeek V3 MoE architecture
            MODEL=$(resolve_model "moonshotai/Moonlight-16B-A3B")
            if [ -z "$MODEL" ]; then echo "SKIP test 7: Moonlight-16B not in cache"; continue; fi
            DIR="$WORK_DIR/t7"; mkdir -p "$DIR"
            write_data_config "$DIR"
            # QLoRA to fit, trust_remote_code for DeepSeek V3
            write_train_config "$DIR" "$MODEL" "load_in_4bit: true
trust_remote_code: true"
            run_test 7 "moonlight-16b-deepseekv3-qlora" \
                "Moonlight 16B-A3B (DeepSeek V3 MoE), QLoRA, single GPU" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        # ---- Feature Tests (using Qwen2-0.5B for speed) ----

        8)  # Liger kernel
            DIR="$WORK_DIR/t8"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$QWEN05B" "use_liger_kernel: true"
            run_test 8 "liger-kernel" \
                "Qwen2-0.5B, LoRA, Liger kernel enabled" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        9)  # Packing (BFD strategy)
            DIR="$WORK_DIR/t9"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$QWEN05B" "packing: true
packing_strategy: bfd"
            run_test 9 "packing-bfd" \
                "Qwen2-0.5B, LoRA, packing with BFD strategy" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        10) # Liger kernel + multi-GPU FSDP
            DIR="$WORK_DIR/t10"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$QWEN05B" "use_liger_kernel: true"
            write_fsdp_config "$DIR"
            run_test 10 "liger-fsdp" \
                "Qwen2-0.5B, LoRA, Liger kernel, FSDP 2 GPUs" \
                "$LOFT train $DIR/train.yaml --accelerate_config $DIR/fsdp.yaml"
            ;;

        11) # Qwen3-VL-2B multimodal + FSDP (multi-GPU multimodal training)
            MODEL=$(resolve_model "Qwen/Qwen3-VL-2B-Instruct")
            if [ -z "$MODEL" ]; then echo "SKIP test 11: Qwen3-VL-2B not in cache"; continue; fi
            DIR="$WORK_DIR/t11"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$MODEL"
            write_fsdp_config "$DIR"
            run_test 11 "qwen3-vl-2b-fsdp" \
                "Qwen3-VL 2B (multimodal), LoRA, FSDP 2 GPUs" \
                "$LOFT train $DIR/train.yaml --accelerate_config $DIR/fsdp.yaml"
            ;;

        12) # Qwen3.5-27B — hybrid linear+full attention, composite model
            MODEL=$(resolve_model "Qwen/Qwen3.5-27B")
            if [ -z "$MODEL" ]; then echo "SKIP test 12: Qwen3.5-27B not in cache"; continue; fi
            DIR="$WORK_DIR/t12"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$MODEL" "load_in_4bit: true
model_parallel: true"
            run_test 12 "qwen35-27b-qlora-mp" \
                "Qwen3.5 27B (hybrid attention), QLoRA, model_parallel across 2 GPUs" \
                "$PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        13) # Qwen3.5-27B + Liger kernel — verifies custom Liger patching for hybrid arch
            MODEL=$(resolve_model "Qwen/Qwen3.5-27B")
            if [ -z "$MODEL" ]; then echo "SKIP test 13: Qwen3.5-27B not in cache"; continue; fi
            DIR="$WORK_DIR/t13"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$MODEL" "load_in_4bit: true
model_parallel: true
use_liger_kernel: true"
            run_test 13 "qwen35-27b-qlora-mp-liger" \
                "Qwen3.5 27B (hybrid attention), QLoRA, model_parallel, Liger kernel" \
                "$PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        14) # Auxiliary losses (all enabled) with Qwen2-0.5B — no Liger
            DIR="$WORK_DIR/t14"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$QWEN05B" "assistant_only_loss: true
aux_loss_eos_weight: 0.1
aux_loss_rep_weight: 0.01
aux_loss_top_prob_weight: 0.01
label_smoothing: 0.1"
            run_test 14 "aux-losses-all" \
                "Qwen2-0.5B, LoRA, all auxiliary losses + label smoothing" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        15) # Auxiliary losses (all enabled) with Qwen2-0.5B + Liger kernel
            DIR="$WORK_DIR/t15"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$QWEN05B" "assistant_only_loss: true
use_liger_kernel: true
aux_loss_eos_weight: 0.1
aux_loss_rep_weight: 0.01
aux_loss_top_prob_weight: 0.01
label_smoothing: 0.1"
            run_test 15 "aux-losses-all-liger" \
                "Qwen2-0.5B, LoRA, all auxiliary losses + label smoothing + Liger" \
                "CUDA_VISIBLE_DEVICES=0 $PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        16) # Qwen3.5-27B + CCE — verifies VL model type dispatch fix
            MODEL=$(resolve_model "Qwen/Qwen3.5-27B")
            if [ -z "$MODEL" ]; then echo "SKIP test 16: Qwen3.5-27B not in cache"; continue; fi
            DIR="$WORK_DIR/t16"; mkdir -p "$DIR"
            write_data_config "$DIR"
            write_train_config "$DIR" "$MODEL" "load_in_4bit: true
model_parallel: true
use_cce: true"
            run_test 16 "qwen35-27b-qlora-mp-cce" \
                "Qwen3.5 27B (hybrid attention), QLoRA, model_parallel, CCE" \
                "$PYTHON -m loft.scripts.sft $DIR/train.yaml"
            ;;

        *)
            echo "Unknown test number: $test_num (valid: 1-16)"
            ;;
    esac
done

# ---- Summary ----
echo ""
echo "================================================================"
echo "ALL TESTS COMPLETE"
echo "Results: $RESULTS_FILE"
echo "Work dir (logs): $WORK_DIR"
echo "================================================================"

{
    echo "================================================================"
    echo "SUMMARY"
    echo "================================================================"
    grep -E "^(TEST|Status:)" "$RESULTS_FILE" | paste - - | sed 's/Status: /=> /'
    echo ""
    echo "Work dir (full logs): $WORK_DIR"
} >> "$RESULTS_FILE"

echo ""
echo "Summary:"
grep -E "^(TEST|Status:)" "$RESULTS_FILE" | paste - - | sed 's/Status: /=> /'
