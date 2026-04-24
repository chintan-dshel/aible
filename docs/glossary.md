---
sidebar_position: 99
title: Glossary
description: Definitions of key terms used throughout this reference, cross-linked to their source pages.
---

# Glossary

Plain-language definitions, one or two sentences each. Every term links to the page where it's treated in full.

---

## A

**Activation function** — A non-linear function applied to a neuron's output before passing it to the next layer. Without it, stacking layers would be equivalent to a single linear transformation and deep networks couldn't learn complex patterns. See [Neural Networks](./foundations/neural-networks).

**Ambient agent** — A background process that monitors a stream of context (calendar, documents, location) and proactively surfaces information or suggestions without requiring the user to initiate. The core engineering problem: deciding when to interrupt vs. stay silent. See [Novel Interaction Paradigms](./aspirational/novel-interaction).

**Approximate Nearest Neighbor (ANN)** — An algorithm that finds vectors close to a query vector without scanning the entire database. Trades a small accuracy loss for dramatically faster search at scale. HNSW and IVF are the two dominant families. See [Vector Databases](./meta-infrastructure/vector-databases).

**ASR (Automatic Speech Recognition)** — The component of a voice pipeline that converts audio to text. Whisper is the dominant open-source option; Deepgram and AssemblyAI are managed alternatives for lower-latency streaming. See [Novel Interaction Paradigms](./aspirational/novel-interaction).

**Attention** — The mechanism that lets each token in a transformer selectively weight every other token when computing its representation. "Self-attention" means the input sequence attends to itself; "cross-attention" means one sequence attends to another (e.g., decoder attending to encoder). See [Attention](./foundations/attention).

**Audit log** — An append-only record of every AI system action: inputs, outputs, model used, latency, user ID, and outcome. Required for compliance, incident investigation, and anomaly detection. See [Audit Logs](./meta-infrastructure/audit-logs).

---

## B

**Backpropagation** — The algorithm that computes how much each weight in a neural network contributed to the error, by propagating the error signal backward from the output through each layer using the chain rule. This is what makes gradient descent tractable for deep networks. See [Neural Networks](./foundations/neural-networks).

**Blue-green deployment** — A deployment pattern that maintains two identical environments (blue and green), with live traffic on one while the other is updated and tested. Switching means redirecting traffic — rollback is instant. See [Deployment Patterns](./production-concerns/deployment-patterns).

**BPE (Byte Pair Encoding)** — The most common tokenization algorithm. Starts with individual characters, then iteratively merges the most frequent adjacent pair into a single token. Produces subword tokens that handle rare words, typos, and multilingual text more gracefully than word-level tokenization. See [Tokenization](./foundations/tokenization).

---

## C

**Calibration** — A model is calibrated if its stated confidence matches its actual accuracy. If a calibrated model says "70% confident," it should be right ~70% of the time. Most LLMs are not well-calibrated by default. See [Confidence Estimation](./production-concerns/confidence-estimation).

**Canary deployment** — A deployment pattern that routes a small fraction of live traffic (1–5%) to a new model or prompt while the old version handles the rest. Limits blast radius if the new version has unexpected failures. See [Deployment Patterns](./production-concerns/deployment-patterns).

**Chain-of-thought (CoT)** — A prompting technique where the model is asked (or trained) to write out its reasoning steps before giving a final answer. Reliably improves accuracy on multi-step reasoning tasks. See [Prompting](./core-building-blocks/prompting).

**Checkpoint (agent)** — A serialized snapshot of an agent's full state — plan, completed steps, accumulated context summary, artifacts — written to disk after each milestone. Enables resumption after a crash and human review at defined intervals. See [Long-Horizon Agents](./aspirational/long-horizon-agents).

**Chunking** — The process of splitting a document into segments before embedding and indexing for retrieval. Chunk size (256–512 tokens is typical) and overlap are the main tuning knobs; both affect retrieval quality significantly. See [RAG](./core-building-blocks/rag).

**Circuit breaker** — A reliability pattern that tracks failure rates and temporarily stops sending requests to a failing service after a threshold is crossed. Prevents cascading failures by allowing the service time to recover. See [Reliability](./production-concerns/reliability).

**Constitutional AI (CAI)** — Anthropic's approach to reducing harmful outputs: define a set of principles (a "constitution"), then train the model to self-critique and revise its outputs against those principles, reducing reliance on human labeling at scale. See [AI Safety Research](./aspirational/ai-safety).

**Context window** — The maximum number of tokens a model can process in a single call — both input and output combined. Content outside the context window is not visible to the model; it has no memory of it. See [How LLMs Work](./foundations/how-llms-work).

---

## D

**Deep learning** — A subset of machine learning that uses neural networks with many layers. The "deep" refers to depth of layers, not sophistication of thought. The layers learn hierarchical representations of the input — early layers detect simple patterns; later layers detect complex compositions of them. See [What is AI?](./foundations/what-is-ai).

**Distillation (knowledge distillation)** — Training a smaller model (student) to mimic the output distribution of a larger model (teacher), rather than training on hard labels. The student learns the teacher's uncertainty and nuance. Phi-3, Gemma, and TinyLlama were built this way. See [Model Distillation](./meta-infrastructure/model-distillation).

---

## E

**Embedding** — A dense vector (list of floats) representing a piece of text, image, or other data in a high-dimensional space where similar items are geometrically close. The foundation of semantic search and RAG. See [Embeddings](./foundations/embeddings).

**Eval (evaluation)** — A systematic test that measures whether an AI system is behaving as intended: correct answers, format compliance, refusal behavior, safety. Without evals, you find out something broke when users report it. See [Evals](./meta-infrastructure/evals).

**Exponential backoff** — A retry strategy where the wait time between attempts grows exponentially (e.g., 1s → 2s → 4s → 8s), typically with added jitter to prevent synchronized retry storms from multiple clients. See [Reliability](./production-concerns/reliability).

---

## F

**Fallback chain** — An ordered list of models to try when the primary model fails or times out. Each model in the chain is tried in sequence; the chain terminates on the first success. See [Fallbacks](./production-concerns/fallbacks).

**Few-shot prompting** — Providing a small number of input-output examples in the prompt to demonstrate the desired behavior without any weight updates. Two to five examples typically saturate the benefit. See [Prompting](./core-building-blocks/prompting).

**Fine-tuning** — Continuing training on a pretrained model using a smaller, task-specific dataset to adapt its behavior. More expensive to update than prompting but can produce more reliable task-specific behavior. See [Training vs. Inference](./foundations/training-vs-inference).

**Function calling (tool use)** — The mechanism by which an LLM signals that it wants to invoke an external function, specifying the function name and arguments. The application executes the function and returns the result; the model uses it to form a final response. See [Function Calling](./core-building-blocks/function-calling).

---

## G

**Graceful degradation** — Designing a system so that failures reduce capability rather than crash the whole application. An AI feature that's unavailable should fall back to a simpler response, not a 500 error. See [Graceful Degradation](./production-concerns/graceful-degradation).

**Gradient descent** — The optimization algorithm that iteratively adjusts model weights in the direction that reduces loss, using the gradient computed by backpropagation. SGD, Adam, and AdamW are the common variants. See [Neural Networks](./foundations/neural-networks).

**Guardrail** — A layer that intercepts inputs or outputs and blocks, modifies, or flags content that violates defined policies. Input guardrails screen prompts before the model sees them; output guardrails screen responses before users see them. See [Guardrails](./meta-infrastructure/guardrails).

**GGUF** — The standard binary format for CPU-optimized quantized models. Introduced by llama.cpp; now the ecosystem standard for running models on consumer hardware without additional dependencies. See [On-Device AI](./aspirational/on-device-ai).

---

## H

**Hallucination** — A confident-sounding model output that is factually incorrect or fabricated. Mechanistically, it arises from the sampling process: the model generates plausible-sounding tokens, not verified facts. Not a bug that can be fully eliminated — only mitigated. See [How LLMs Work](./foundations/how-llms-work).

---

## I

**Indirect prompt injection** — An attack where malicious instructions are embedded in content the model retrieves or processes (web pages, documents, tool results) rather than in the user's direct input. Particularly dangerous in RAG and agentic systems. See [Prompt Injection](./meta-infrastructure/prompt-injection).

---

## J

**Jitter** — Randomness added to retry wait times to prevent multiple clients from retrying simultaneously and amplifying load on an already-stressed service. Typically implemented as `wait = base_wait * (1 + random(0, 0.5))`. See [Reliability](./production-concerns/reliability).

---

## K

**KV cache** — A cache of the key and value matrices computed for prior tokens during autoregressive generation. Allows the model to generate new tokens without recomputing attention over the entire preceding context, making generation much faster. See [How LLMs Work](./foundations/how-llms-work).

---

## L

**LLM-as-judge** — Using a language model to evaluate the output of another language model, scoring it against qualitative rubrics. Cheaper than human labeling at scale; requires calibration against human judgments to ensure the judge is reliable. See [Evals](./meta-infrastructure/evals).

**Logits** — The raw, unnormalized scores a model outputs before the softmax converts them to probabilities. A higher logit means a more likely next token. See [How LLMs Work](./foundations/how-llms-work).

**Long-horizon agent** — An agent that runs for hours or days, handling tasks too large for a single context window — with checkpointing, context compression, human-in-the-loop gates for irreversible actions, and error recovery. See [Long-Horizon Agents](./aspirational/long-horizon-agents).

---

## M

**MCP (Model Context Protocol)** — Anthropic's open protocol for connecting LLMs to external tools and data sources via a standardized client-server interface. Without MCP, every tool integration is bespoke; MCP standardizes the interface so any MCP-compatible tool works with any MCP-compatible host. Servers expose tools; hosts (Claude Desktop, IDEs) aggregate multiple servers and present them to the model. See [MCP](./meta-infrastructure/mcp).

**Mechanistic interpretability** — A research program that reverse-engineers what's happening inside a model at the level of individual circuits and features. Anthropic's work on superposition and circuits is the leading program; it currently explains tiny fractions of frontier model behavior. See [AI Safety Research](./aspirational/ai-safety).

**Memory (agent)** — How an agent stores and retrieves information across turns. Four types: in-context (in the prompt), external (vector DB, key-value store), episodic (past interaction log), and parametric (baked into weights via fine-tuning). See [Memory Architectures](./core-building-blocks/memory-architectures).

**Model routing** — Directing requests to different models based on complexity, cost, or capability requirements — sending simple queries to a fast cheap model and complex queries to a capable expensive one. See [Model Routing](./meta-infrastructure/model-routing).

**Multi-agent system** — An architecture where multiple AI agents collaborate: orchestrators that plan and delegate, subagents that execute specialized tasks. Key design questions: trust, how agents communicate, and how errors propagate. See [Multi-Agent Systems](./core-building-blocks/multi-agent-systems).

**Multi-head attention** — Running multiple attention operations in parallel, each with its own learned projections. Different heads learn to attend to different relationships — one head might track syntax while another tracks coreference. Their outputs are concatenated and projected. See [Attention](./foundations/attention).

---

## O

**Observability** — The ability to understand what your AI system is doing from the outside, by examining its outputs. In LLM systems: tracing individual requests end-to-end, tracking latency, cost, and quality metrics, and correlating failures to their causes. See [Observability](./meta-infrastructure/observability).

**On-device inference** — Running model inference on local hardware (phone, laptop, embedded system) rather than sending data to a cloud API. Eliminates latency, cost, and privacy concerns; constrained by model capability and hardware limits. See [On-Device AI](./aspirational/on-device-ai).

---

## P

**Pretraining** — The first phase of LLM training: train on a massive corpus of text to predict the next token. This is where the model acquires its capabilities and world knowledge. Fine-tuning and RLHF happen afterward. See [Training vs. Inference](./foundations/training-vs-inference).

**Prompt** — The input sent to a language model. A prompt consists of some combination of: a system prompt (instructions and persona), a conversation history, and a user message. The model generates a completion. See [How LLMs Work](./foundations/how-llms-work).

**Prompt injection** — An attack where a malicious user (or content the model processes) embeds instructions that attempt to override the system prompt or make the model take unintended actions. See [Prompt Injection](./meta-infrastructure/prompt-injection).

---

## Q

**Quantization** — Reducing the numerical precision of model weights from float32 (4 bytes/weight) to int8 (1 byte) or int4 (0.5 bytes). Makes models small enough to run on consumer hardware at the cost of some quality degradation — minimal at 8-bit, noticeable at 4-bit on reasoning tasks. See [On-Device AI](./aspirational/on-device-ai).

---

## R

**RAG (Retrieval-Augmented Generation)** — A pattern that retrieves relevant documents from an external knowledge base at query time and includes them in the model's context, grounding the response in specific content rather than the model's training data. See [RAG](./core-building-blocks/rag).

**Red teaming** — Adversarial testing where you actively try to make an AI system fail — through jailbreaks, edge cases, boundary conditions, and prompt attacks — before a real attacker does. See [Red Teaming](./meta-infrastructure/red-teaming).

**RLHF (Reinforcement Learning from Human Feedback)** — A training approach that fine-tunes a pretrained model using human preference data: a reward model is trained on human comparisons, then the LLM is trained to maximize the reward model's score. The current mainstream alignment technique. See [Training vs. Inference](./foundations/training-vs-inference).

**RSSM (Recurrent State Space Model)** — The core learned world model in Dreamer-style model-based RL. Encodes observations into a compact latent state and learns to predict next states in imagination — letting the agent plan entire action sequences without any environment calls. This is what makes Dreamer-style RL practical at scale. See [World Models](./aspirational/world-models).

---

## S

**Self-attention** — Attention applied within a single sequence: each token attends to every other token in the same input. This is what allows transformers to capture long-range dependencies without RNN-style sequential processing. See [Attention](./foundations/attention).

**Self-consistency** — A confidence estimation technique that samples multiple outputs for the same prompt and measures agreement. High agreement → higher confidence; low agreement → lower confidence. Costs N× the API budget. See [Confidence Estimation](./production-concerns/confidence-estimation).

**Semantic cache** — A cache that returns stored responses for queries that are semantically similar (not just exactly identical) to a cached query, using embedding-based similarity. Saves cost and latency for paraphrase variations of the same request. See [Caching](./meta-infrastructure/caching).

**Shadow mode** — Running a new model or prompt in parallel with the production model, without serving its output to users. Used to collect comparison data before a live rollout. See [Deployment Patterns](./production-concerns/deployment-patterns).

**Structured output** — Model output constrained to a specific format (usually JSON matching a schema) rather than free text. Produced via constrained decoding, JSON mode, or tool-schema-enforced generation. See [Structured Outputs](./core-building-blocks/structured-outputs).

**Superposition** — A property of neural networks where individual neurons participate in representing multiple distinct features simultaneously. Features are directions in activation space, not one-to-one with neurons. This is why reading a model's behavior from individual neurons is misleading. See [AI Safety Research](./aspirational/ai-safety).

**Synthetic data** — Training or evaluation data generated by an AI model rather than collected from the real world. Useful for augmenting scarce datasets and testing edge cases; risks: model-specific biases baked in, degradation if used uncritically for self-training. See [Synthetic Data](./meta-infrastructure/synthetic-data).

---

## T

**Temperature** — A sampling parameter that controls randomness in token selection. Temperature 0 selects the highest-probability token deterministically; temperature 1 samples from the raw distribution; values above 1 increase randomness further. See [How LLMs Work](./foundations/how-llms-work).

**Token** — The basic unit a language model processes. Not a word — typically a word fragment: "unhappy" might be "un" + "happy". The average English word is ~1.3 tokens. See [Tokenization](./foundations/tokenization).

**Top-p (nucleus sampling)** — A sampling strategy that considers only the smallest set of tokens whose cumulative probability exceeds p, then samples from that set. Dynamically adjusts how many tokens are considered based on the distribution's shape. See [How LLMs Work](./foundations/how-llms-work).

**Transformer** — The neural network architecture underlying all modern LLMs. Key components: multi-head self-attention (to relate tokens to each other), feed-forward layers (per-token computation), and residual connections (to stabilize training at depth). See [Transformers](./foundations/transformers).

**TTS (Text-to-Speech)** — The component of a voice pipeline that converts text to audio. ElevenLabs produces high-quality neural voices; pyttsx3 is a cross-platform offline option with lower quality; OpenAI's Realtime API collapses the TTS step into an end-to-end model. See [Novel Interaction Paradigms](./aspirational/novel-interaction).

---

## V

**VAD (Voice Activity Detection)** — A component that identifies segments of an audio stream that contain speech, filtering out silence and background noise before passing audio to the ASR model. Reduces cost and error rates significantly in noisy environments. See [Novel Interaction Paradigms](./aspirational/novel-interaction).

**Vector database** — A database optimized for storing and querying high-dimensional embedding vectors by approximate nearest-neighbor similarity. Pinecone, Qdrant, Weaviate, and ChromaDB are common options; each makes different tradeoffs on managed vs. self-hosted, scale, and filtering. See [Vector Databases](./meta-infrastructure/vector-databases).

---

## W

**World model** — A system's internal representation of how reality works — one that lets it predict the consequences of actions without taking them. In model-based RL (Dreamer), an explicit learned world model. Whether LLMs have implicit world models is actively debated. See [World Models](./aspirational/world-models).
