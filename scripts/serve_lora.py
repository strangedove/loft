#!/usr/bin/env python3
"""Lightweight OpenAI-compatible server for Qwen3.5 + PEFT LoRA adapters.

Supports dynamic adapter selection via the `model` field in chat completions.
"""

import argparse
import json
import time
import uuid
from threading import Lock

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
lock = Lock()

# Global state
model = None
tokenizer = None
adapters = {}  # name -> path
active_adapter = None
base_model_name = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "base"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "base"
    prompt: str | list[str] = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False
    stop: list[str] | str | None = None


def switch_adapter(name: str):
    global model, active_adapter
    if name == "base":
        if active_adapter is not None:
            model.disable_adapter_layers()
            active_adapter = None
        return
    if name not in adapters:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(adapters.keys()) + ['base']}")
    if active_adapter is None:
        model.enable_adapter_layers()
    model.set_adapter(name)
    active_adapter = name


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    models = [{"id": "base", "object": "model"}]
    for name in adapters:
        models.append({"id": name, "object": "model"})
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    with lock:
        switch_adapter(request.model)

        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature if request.temperature > 0 else None,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
            "repetition_penalty": request.repetition_penalty,
        }
        if request.min_p > 0:
            generate_kwargs["min_p"] = request.min_p

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)
        t1 = time.time()

        new_tokens = output[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = len(new_tokens)

        # Check for thinking tags
        full_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        reasoning_content = None
        if "<think>" in full_text:
            parts = full_text.split("</think>", 1)
            if len(parts) == 2:
                reasoning_content = parts[0].replace("<think>", "").strip()
                text = parts[1].strip()
                # Remove any trailing special tokens
                for tok in ["<|im_end|>", "<|endoftext|>"]:
                    text = text.replace(tok, "").strip()
                    if reasoning_content:
                        reasoning_content = reasoning_content.replace(tok, "").strip()

        message = {"role": "assistant", "content": text}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop" if completion_tokens < request.max_tokens else "length",
                }
            ],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": completion_tokens,
                "total_tokens": input_len + completion_tokens,
            },
            "timings": {
                "total_s": round(t1 - t0, 3),
                "tokens_per_second": round(completion_tokens / (t1 - t0), 1) if t1 > t0 else 0,
            },
        }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    with lock:
        switch_adapter(request.model)

        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        stop_strings = []
        if request.stop:
            stop_strings = [request.stop] if isinstance(request.stop, str) else request.stop

        generate_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature if request.temperature > 0 else None,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
            "repetition_penalty": request.repetition_penalty,
        }
        if request.min_p > 0:
            generate_kwargs["min_p"] = request.min_p

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)
        t1 = time.time()

        new_tokens = output[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Handle stop strings
        finish_reason = "length" if len(new_tokens) >= request.max_tokens else "stop"
        for s in stop_strings:
            if s in text:
                text = text[:text.index(s)]
                finish_reason = "stop"
                break
        # Clean trailing special tokens
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            text = text.replace(tok, "")

        completion_tokens = len(new_tokens)

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": completion_tokens,
                "total_tokens": input_len + completion_tokens,
            },
            "timings": {
                "total_s": round(t1 - t0, 3),
                "tokens_per_second": round(completion_tokens / (t1 - t0), 1) if t1 > t0 else 0,
            },
        }


def main():
    global model, tokenizer, adapters, base_model_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--lora-modules",
        nargs="+",
        help="LoRA modules in name=path format",
        default=[],
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit (QLoRA-compatible)")
    args = parser.parse_args()

    base_model_name = args.model
    dtype = getattr(torch, args.dtype)

    print(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.lora_modules:
        lora_specs = {}
        for spec in args.lora_modules:
            name, path = spec.split("=", 1)
            lora_specs[name] = path

        # Load first adapter
        first_name = list(lora_specs.keys())[0]
        first_path = lora_specs[first_name]
        print(f"Loading LoRA adapter: {first_name} from {first_path}")
        model = PeftModel.from_pretrained(model, first_path, adapter_name=first_name)
        adapters[first_name] = first_path

        # Load remaining adapters
        for name, path in lora_specs.items():
            if name == first_name:
                continue
            print(f"Loading LoRA adapter: {name} from {path}")
            model.load_adapter(path, adapter_name=name)
            adapters[name] = path

        # Start with adapters disabled (base model)
        model.disable_adapter_layers()

    model.eval()
    print(f"Model loaded. Available models: base, {', '.join(adapters.keys())}")
    print(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
