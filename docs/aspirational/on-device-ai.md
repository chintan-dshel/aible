---
sidebar_position: 6
title: On-Device AI
description: Quantization, edge inference, and the privacy and latency implications of running models locally.
---

# On-Device AI

Running language models and ML inference on local hardware — phones, laptops, embedded systems — rather than sending data to a cloud API.

## The problem it solves

Cloud inference has four friction points: latency (round-trip to a datacenter), cost (at scale, per-token pricing adds up), privacy (user data leaves the device), and availability (network required). On-device inference eliminates all four — but trades them for model capability constraints, hardware limits, and deployment complexity.

The inflection point happened around 2023–2024. Models like Phi-3 Mini (3.8B), Gemma 2 (2B), and Mistral 7B showed that a model small enough to run on a laptop or phone could handle a surprising range of tasks: summarization, classification, code completion, simple QA, and local RAG — acceptably well, not just barely.

## How it works under the hood

**Quantization.** The dominant technique. Reduce the precision of model weights from float32 (4 bytes/weight) to float16 (2 bytes), int8 (1 byte), or int4 (0.5 bytes). A 7B-parameter model in float32 = 28 GB. In 4-bit = 3.5 GB — fits in a phone's RAM. The tradeoff: lower precision degrades quality, especially on reasoning tasks. 8-bit is nearly lossless for most tasks; 4-bit is noticeably worse on multi-step reasoning but acceptable for classification, summarization, and retrieval.

**GGUF format.** The standard format for CPU-optimized quantized models. llama.cpp introduced it; now the ecosystem standard for running models on consumer hardware. A GGUF file contains the quantized weights, metadata, and tokenizer — everything needed to run the model without additional dependencies.

**Hardware acceleration.** Modern consumer hardware has specialized inference units: Apple Neural Engine (ANE) on M-series chips, Qualcomm Hexagon NPU on Snapdragon, Intel Neural Compute Stick. These provide 5–20× speedup over pure CPU inference. Frameworks like CoreML (Apple), ONNX Runtime, and MLC LLM compile models to target these units.

**Knowledge distillation.** Separately from quantization: train a smaller model (student) to mimic the outputs of a larger model (teacher). The student learns from soft labels — the teacher's full probability distribution over all possible next tokens, rather than just a one-hot correct answer (the "hard label"). This transfers the teacher's uncertainty and nuance, not just its top prediction. See [[Model Distillation]] for the full training recipe. Phi-3, Gemma, and TinyLlama are all distilled from larger models.

**Speculative decoding on edge.** Use a tiny draft model (50M–500M params) to generate candidate tokens, then verify with the full model. On-device, where memory bandwidth is the bottleneck, this can 2–4× decode speed.

## Concrete example

```python
# Running a local model with llama-cpp-python
# Install: pip install llama-cpp-python
# Download a specific GGUF variant: huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir ./models

from llama_cpp import Llama
from pathlib import Path
from typing import Callable


def load_local_model(
    model_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,  # -1 = use all available GPU layers
    verbose: bool = False,
) -> Llama:
    """
    Load a GGUF model for local inference.
    n_gpu_layers=-1 uses GPU (Metal on Mac, CUDA on NVIDIA) if available,
    falls back to CPU automatically.
    """
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )


def local_completion(
    llm: Llama,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.1,  # low temperature for determinism
    stop: list[str] | None = None,
) -> str:
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or ["<|end|>", "</s>", "[INST]"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def local_chat(
    llm: Llama,
    messages: list[dict],
    max_tokens: int = 512,
) -> str:
    """Chat completion using the model's native chat template."""
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return output["choices"][0]["message"]["content"]


def privacy_sensitive_classification(
    llm: Llama,
    text: str,
    categories: list[str],
) -> str | None:
    """
    Classify sensitive text entirely on-device — no data leaves the machine via
    network calls. Note: data is still accessible to local processes and system logs;
    "on-device" is not a substitute for OS-level access controls.
    Useful for health, legal, or financial documents.
    """
    category_list = "\n".join(f"- {c}" for c in categories)
    prompt = (
        f"Classify the following text into exactly one of these categories:\n"
        f"{category_list}\n\n"
        f"Text: {text}\n\n"
        f"Category (reply with only the category name):"
    )
    result = local_completion(llm, prompt, max_tokens=20)
    # Validate result is one of the expected categories
    for category in categories:
        if category.lower() in result.lower():
            return category
    return None  # model output didn't match any expected category — caller must handle


# Benchmarking on-device vs cloud
import time


def benchmark_inference(llm: Llama, prompt: str, n_runs: int = 5) -> dict:
    times = []
    token_counts = []

    for _ in range(n_runs):
        start = time.time()
        output = llm(prompt, max_tokens=128, echo=False)
        elapsed = time.time() - start
        tokens = output["usage"]["completion_tokens"]
        times.append(elapsed)
        token_counts.append(tokens)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)

    return {
        "avg_latency_s": round(avg_time, 2),
        "avg_tokens": round(avg_tokens, 1),
        "tokens_per_second": round(avg_tokens / avg_time, 1),
        "runs": n_runs,
    }


# Example: local private document assistant
def build_offline_assistant(model_path: str) -> Callable:
    llm = load_local_model(model_path, n_ctx=8192, verbose=False)

    def ask(question: str, context: str = "") -> str:
        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"Answer questions based on this context:\n\n{context}",
            })
        messages.append({"role": "user", "content": question})
        return local_chat(llm, messages)

    return ask
```

On a MacBook M2 Pro, Phi-3 Mini 4K (Q4_K_M quantization) runs at ~40 tokens/second — fast enough for interactive use. On a Raspberry Pi 5, the same model runs at ~3 tokens/second — usable for batch processing, not for real-time chat.

## When to use it / when not to

**Use when:**
- Data must not leave the device via network (health records, legal documents, personal journals, confidential IP) — note that "on-device" means no API calls, not that the data is inaccessible to other local processes or system logs. For strict isolation, OS-level access controls (app sandboxing, file permissions) are also required.
- Offline capability is required (field work, air-gapped systems, unreliable connectivity)
- High-volume batch processing where cloud per-token cost is prohibitive
- Latency requirements are under 100ms for short completions (consumer hardware can hit this with small models)

**Don't use when:**
- Task requires a frontier model's reasoning capability — 7B models are significantly worse than Claude Sonnet or GPT-4o on complex multi-step reasoning, code generation, and instruction following
- Context window > 8K tokens — small models degrade sharply at long contexts
- Frequent model updates are needed — deploying a new model version to 10,000 devices is an operational challenge cloud models don't have
- Hardware is constrained below 4 GB RAM — even quantized 7B models need 4–6 GB

## Main tools and libraries

| Tool | Role |
|------|------|
| `llama-cpp-python` | Python bindings for llama.cpp — runs GGUF models on CPU/GPU/Metal |
| Ollama | Easiest way to run models locally — one-command download and serve |
| MLC LLM | High-performance inference on phones (iOS, Android) and desktops |
| CoreML Tools | Converts models to Apple's native format for ANE acceleration |
| ONNX Runtime | Cross-platform inference — Windows, Linux, Android, iOS |
| Hugging Face Transformers + `bitsandbytes` | Load quantized models in Python (8-bit, 4-bit via BitsAndBytes) |

**Recommended models by use case:**
- General assistant / chat: Phi-3 Mini (3.8B), Gemma 2 2B, Mistral 7B
- Code completion: CodeLlama 7B, DeepSeek Coder 6.7B
- Embedding (for local RAG): nomic-embed-text, all-MiniLM-L6-v2

## Common failure modes and gotchas

**Quantization quality loss on reasoning tasks.** 4-bit quantization is nearly invisible on classification and summarization but noticeably degrades multi-step arithmetic, complex instruction following, and long-context reasoning. Test your specific task at each quantization level before assuming 4-bit is acceptable.

**Thermal throttling.** Sustained inference on a phone or laptop generates heat. After 2–5 minutes of continuous generation, the device throttles CPU/GPU clock speeds and tokens-per-second drops 30–60%. Design for bursty use or add cooling headroom.

**Context window inflation on small models.** A 7B model with a 4K context window degrades in coherence near the end of the context. Budget context more conservatively than with frontier models.

**Memory spikes during model loading.** Loading a 7B GGUF model requires the full model in RAM simultaneously with the KV cache. On devices with 8 GB RAM, this leaves little headroom for the OS and other apps. Monitor peak RAM, not just steady-state.

**Model drift from cloud version.** "Phi-3 Mini on-device" and "GPT-4o via API" are not interchangeable. Prompts tuned for one often don't transfer to the other. Maintain separate prompt templates and evals for each deployment target.

## Project ideas

- **Private document assistant**: build a local RAG system using Ollama + nomic-embed-text + a local GGUF model. Documents are embedded locally, stored in a local vector store (ChromaDB with persistent storage), and queried entirely on-device. See [[RAG]] for retrieval patterns and [[Vector Databases]] for local store options.
- **On-device classification service**: wrap a local model in a FastAPI server on localhost — no network calls, sub-100ms classification for on-device applications (macOS menu bar app, iOS Shortcut, VS Code extension).
- **Quantization quality benchmark**: take 50 representative prompts from your use case, run them through Q8_0, Q6_K, Q4_K_M, and Q3_K_S variants of the same model, and measure output quality degradation vs. tokens/second tradeoff. The result tells you which quantization level is the right tradeoff for your task.

## Going deeper

- [llama.cpp repository](https://github.com/ggerganov/llama.cpp) — the canonical CPU inference engine; GGUF format docs
- [Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314) — quantization for fine-tuning; covers quantization fundamentals
- [MLC LLM documentation](https://llm.mlc.ai/) — mobile and edge deployment
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) — Microsoft's approach to small, capable models
- [[Model Distillation]] — how small capable models like Phi-3 are created from larger teacher models
- [[Latency Optimization]] — streaming, batching, and speculative decoding for faster inference
