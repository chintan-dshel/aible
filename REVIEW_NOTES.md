# REVIEW_NOTES.md

Minor issues flagged by reviewers that don't block phase completion.
Critical and important issues are addressed before phases close.

---

## Phase 1 review — 2026-04-23 (what-is-ai.md, attention.md)

### Deferred (address when relevant pages are written)

**[what-is-ai.md] XGBoost caveats** — Red Team noted that "XGBoost often wins on tabular data" omits cases where tabular transformers (TabNet, FT-Transformer) are competitive on large dense-feature datasets. Adding full caveats would over-engineer the page. Revisit in a dedicated tabular-ML page.

**[attention.md] RNNs still live in embedded/streaming** — Red Team noted "most language tasks" overstates transformer dominance; RNNs survive in hearing aids, microcontrollers, streaming audio. The "from 2018 onward" fix softens this. If an embedded-AI page is written, note the exception there.

**[attention.md] Attention ≠ Explanation — one-sided citation** — Red Team flagged that Wiegreffe & Pinter (2019, "Attention is not not Explanation") partially rebuts Jain & Wallace, and the consensus is more nuanced. Add the rebuttal citation when this failure mode section is revisited.

**[attention.md] FlashAttention "memory > FLOPs" lesson scope** — Red Team noted the lesson is attention-specific; for MLP layers, FLOP utilization is often the bottleneck. Add a parenthetical scoping note to the My take callout: "this pattern applies to attention specifically — for dense MLP layers the bottleneck is usually raw FLOP utilization."

**[attention.md] Library analogy self-attention gap** — Pedagogical reviewer: the library analogy breaks at the point where Q, K, V all come from the same sequence. Add one sentence explaining why self-attention queries itself: each token gathers context from every other position in the same sequence.

**[attention.md, Project idea #3] Winograd Schema source** — Pedagogical reviewer: "Winograd Schema" is jargon; add the dataset source. Hugging Face: `winogrande`, or the canonical set at cs.nyu.edu/~davise/Winograd/WS.html.

**[what-is-ai.md] "Intermediate representations" order** — Pedagogical reviewer: the term appears in the ML section before it's explained in the Deep Learning section. Consider moving the explanation up or deferring the term.

---

## Phase 2 review — 2026-04-23 (8 new pages: neural-networks, history-timeline, history-narrative, transformers, embeddings, tokenization, training-vs-inference, how-llms-work)

Two review rounds completed (round 1 in prior session, round 2 this session). All must-fix and fix-in-current-pass items applied before phase close. Items below are deferred.

### Round 2 deferred — Pedagogical (add intuition / expand)

**[neural-networks.md] Perceptron diagram caption** — Pedagogical: "Σ wᵢxᵢ + b" label could be read as the activation itself. Add a note that this is the pre-activation, which then passes through σ.

**[neural-networks.md] Activation function table — "saturates/vanishing gradients" jargon** — Pedagogical: readers without calculus background won't know what "saturates" or "vanishing gradient" means on first encounter. Add a one-sentence preamble before the table.

**[neural-networks.md] Weight initialization — no intuition for scaling factors** — Pedagogical: the Xavier and He formulas are stated without explaining why those specific values keep activation variance stable. Add a sentence: "If weights start too large, activations explode; too small, they vanish. The scaling factors keep the average activation variance constant across layers."

**[neural-networks.md] Residual connections — gradient flow prose** — Pedagogical: the skip path benefit is stated but the mechanism ("why does skipping help?") isn't spelled out. Add: "In networks without residuals, gradients must traverse every layer's multiplication; deep stacking causes them to shrink. The skip path is a gradient shortcut."

**[history-narrative.md] SVM training finickiness** — Pedagogical: "neural networks required finicky training" is stated without explaining what made them finicky. Add one sentence: "Neural networks required careful tuning of learning rates, initialization, and architecture; even small hyperparameter errors could prevent training from converging at all."

**[history-narrative.md] Few-shot learning definition** — Pedagogical: "few-shot learning" is used without definition in the narrative. Add: "(providing a few input-output examples in the prompt, with no weight updates)."

**[transformers.md] RNN information bottleneck** — Pedagogical: "information bottleneck" assumes familiarity with sequential hidden states. Add: "In RNNs, the model encodes the entire past into a fixed-size state vector. If context is long, early information fades. Attention lets each token directly access all positions, bypassing this compression."

**[transformers.md] Positional encoding choice criteria** — Pedagogical: three PE schemes listed with no guidance on when to choose each. Add: "Sinusoidal/learned for projects with context ≤ 4K. RoPE for modern LLMs needing longer context. ALiBi if reliable generalization beyond training length is required without retraining."

**[transformers.md] Decoder-only understanding mechanism** — Pedagogical: "surprisingly good at understanding tasks via prompting" doesn't explain why next-token prediction trains understanding. Add: "To predict 'Paris' after 'capital of France is,' the model must represent that relationship. Understanding is a byproduct."

**[transformers.md] Attention sink definition** — Pedagogical: "attention sinks" in failure modes is jargon without definition. Add: "Attention sinks: high attention weight assigned to [BOS] or padding tokens, wasting capacity. Result: effective context is shorter than the model's technical limit."

**[embeddings.md] One-hot similarity gap** — Pedagogical: "every word equally distant from every other" is stated but the practical consequence isn't immediate. Add: "The algorithm can't tell 'dog' is closer to 'puppy' than to 'door.' Dense embeddings make distance meaningful."

**[embeddings.md] Skip-gram equation — no intuition preamble** — Pedagogical: the equation appears without explaining what it's doing. Add before: "Skip-gram trains by prediction: given 'cat,' predict nearby words. Words appearing in similar contexts develop similar vectors. The equation maximizes this prediction probability."

**[embeddings.md] Contrastive loss notation** — Pedagogical: $\tau$, $p^+$, $p_i^-$ not explained in plain language before the equation. Add: "Contrastive loss pulls query embeddings toward relevant passages (p+) and pushes away from irrelevant ones (p-). Temperature τ controls how sharp this pull-and-push is."

**[embeddings.md] Cosine similarity ≠ relevance — position** — Pedagogical: this warning appears in failure modes but not at point-of-use (the "What it is" section). Add a note early: "Geometric closeness is a useful proxy for relevance, but not identical to it — see failure mode #6."

**[embeddings.md] Re-ranking definition** — Pedagogical: "re-ranking" is mentioned in failure modes without defining it. Add: "Re-ranking: use a cross-encoder model to directly score (query, document) pairs, catching semantic false positives that embedding similarity misses."

**[tokenization.md] 5× character-level inefficiency figure** — Pedagogical: "5× as many tokens" needs grounding. Add: "A 10-word English sentence is ~50 characters. Character-level = 50 tokens; word-level = ~10 tokens; BPE = ~12 tokens."

**[tokenization.md] WordPiece likelihood maximization** — Pedagogical: "maximize corpus likelihood under the language model" is too abstract. Simplify to: "WordPiece considers which merges help predict the training text better, not just which pairs appear most often."

**[tokenization.md] Vocabulary table implications** — Pedagogical: embedding table memory size is shown but the practical implication isn't. Add: "At inference, the embedding table must fit in GPU memory; 1GB on a 24GB GPU is 4% of VRAM — manageable, but notable at scale."

**[training-vs-inference.md] Pretraining → alignment motivation** — Pedagogical: the jump from "predicts next token" to "why alignment is necessary" isn't made explicit. Add: "This is why alignment training is necessary — the next-token objective doesn't distinguish safe from unsafe completions."

**[training-vs-inference.md] DPO intuition before equation** — Pedagogical: the DPO loss equation appears without plain-language preamble. Add before: "DPO skips the reward model and trains directly: maximize probability of preferred responses while staying close to the reference model weights."

**[training-vs-inference.md] Low-rank adaptation intuition** — Pedagogical: why low-rank works isn't explained. Add: "Most weight updates needed for fine-tuning occupy a lower-dimensional subspace — you don't need to update all parameters. Rank controls this subspace: rank 4 for light adaptation, rank 64 for deeper changes."

**[training-vs-inference.md] Catastrophic forgetting severity guidance** — Pedagogical: the failure mode is described but it's not clear when it matters vs. when it's acceptable. Add: "Catastrophic forgetting is only a problem if you need the model to remain capable on general tasks. If your use case is narrowly scoped, forgetting is fine."

**[how-llms-work.md] KV cache and pipeline feedback loop** — Pedagogical: the loop diagram doesn't call out where KV caching applies. Add a caption note tying the loop to the cache.

**[how-llms-work.md] Top-p numeric example** — Pedagogical: the top-p description is correct but abstract. Add: "E.g., probabilities [0.7, 0.15, 0.08, ...] with p=0.9: keep the first three (cumulative 0.93 > 0.9), sample from them renormalized."

**[how-llms-work.md] Context window as memory — implications** — Pedagogical: the "no memory between conversations" point is made but the corollary (context is never compressed into weights mid-session) isn't. Add: "Old messages don't 'compress' into parameters — they get truncated when context fills."

**[how-llms-work.md] Hallucination mitigation** — Pedagogical: hallucination is described mechanistically but fixes aren't given. Add: "Mitigations: lower temperature, use RAG to bias probabilities toward retrieved facts, add a verifier model. None eliminates hallucinations entirely."

### Round 2 deferred — Red Team (framing / completeness)

**[history-narrative.md] Symbolic AI failure framing** — Red Team: "failed because encoding knowledge by hand doesn't scale" undersells that expert systems worked well on narrow tasks. Clarify: "succeeded on narrow, well-defined problems but proved brittle and expensive to maintain as domains evolved."

**[history-narrative.md] Reasoning vs. pattern matching framing** — Red Team: "a question philosophers of mind argue about" sidesteps the empirical evidence. Expand: "…is contested. Some evidence suggests models memorize patterns at scale; other work documents genuine compositional reasoning. The mechanism remains unclear."

**[neural-networks.md] Code robustness — device handling** — Red Team: The device fix was applied (`.to(device)`), but the training loop still moves data to device implicitly. Add `X, y = X.to(device), y.to(device)` inside the training loop.

**[training-vs-inference.md] Constitutional AI principles** — Red Team: CAI described without what a "constitution" actually contains. Add: "The 'constitution' is a set of principles (e.g., 'be helpful, harmless, and honest') the model uses to critique and revise its own outputs."

**[training-vs-inference.md] Code — HF Hub access gate** — Red Team: the QLoRA example loads Llama 3 without noting that access requires accepting terms on HuggingFace Hub. Add a comment.

**[training-vs-inference.md] LoRA rank guidance — knowledge injection** — Red Team: "knowledge injection" via LoRA is misleading; LoRA adapts existing capabilities, it doesn't inject new facts. Revise: "Rank 16–64 for task-specific capability adaptation. LoRA does not inject new factual knowledge — use continued pretraining for that."

**[how-llms-work.md] "Compression artifacts" metaphor scope** — Red Team: the My Take compression analogy is useful but should be flagged as metaphorical. Add: "This is one useful analogy; the actual mechanisms by which models generalize remain an active research question."

**[how-llms-work.md] Hallucination — multiple causes** — Red Team: "sampling artifact" is reductive. Hallucination also occurs from contradictory training data, OOD inputs, and miscalibrated confidence. Expand the definition.

**[how-llms-work.md] KV cache — memory as bottleneck** — Red Team: the KV cache section correctly notes linear growth but doesn't flag that memory (not compute) is the primary bottleneck for long-context inference. Add explicitly.

### Deferred (address when relevant pages are written or on next pass)

**[neural-networks.md] XOR binary classification vs loss function** — Red Team: the XOR table implies binary 0/1 outputs but the training loop uses CrossEntropyLoss over 2 classes. This is correct but potentially confusing; the two framings aren't reconciled. Add a note if a dedicated loss-functions page is written.

**[neural-networks.md] "Universal approximation" scope** — Red Team: the UAT as stated applies to a single hidden layer of sufficient width; depth universality requires different conditions. Narrowing to "with sufficient width" would be more precise. Low priority — the distinction doesn't change practical guidance.

**[neural-networks.md] BatchNorm eval mode vs training mode gap** — Pedagogical reviewer: the BatchNorm description doesn't mention `model.eval()` switching to running statistics. The concrete example does show `model.eval()` correctly; adding a one-sentence explanation of what changes at eval time would close the gap.

**[neural-networks.md] Dead ReLU 20–50% figure** — Red Team: the "can silently kill 20–50% of neurons" figure is architecture- and initialization-dependent; no citation. Soften to "can kill a significant fraction of neurons in under-tuned configurations" if a citation can't be located.

**[history-narrative.md] ELIZA anthropomorphization anecdote** — Pedagogical reviewer: the narrative focuses on researcher surprise but omits that many users consciously knew ELIZA was a chatbot yet still confided in it. The distinction matters for modern discussions of human-AI attachment. Add if the history section is expanded.

**[history-narrative.md] AlphaGo vs AlphaZero distinction** — Red Team: the narrative conflates AlphaGo (human games + self-play, 2016) with AlphaZero (self-play only, 2017). Currently only the timeline mentions AlphaGo; the narrative doesn't cover it explicitly. If the Games section expands, distinguish the two.

**[transformers.md] Sinusoidal PE — learned vs fixed framing** — Red Team: the page states sinusoidal PE "generalizes to unseen sequence lengths" but doesn't note that in practice, most modern models use learned PE capped at max training length — sinusoidal extrapolation is theoretically possible but rarely used in production. Add a parenthetical if the positional encoding section is expanded.

**[transformers.md] T5 bias framing** — Pedagogical reviewer: T5 is listed under "encoder-decoder" but T5's framing (everything as a text-to-text problem) is architecturally notable — it's not just seq-to-seq. Add a sentence on the text-to-text framing if an encoder-decoder deep-dive page is written.

**[transformers.md] $d_{\text{ff}} = 4 \times d_{\text{model}}$ varies in modern architectures** — Red Team: the $4\times$ rule is from the original transformer and early GPT models; SwiGLU variants (Llama, PaLM) use different ratios. Add a caveat if the FFN section is expanded.

**[embeddings.md] MTEB leaderboard staleness** — Red Team: MTEB rankings change monthly. The My Take callout refers to "the current MTEB leaderboard" — add a date stamp or caveat that rankings shift. Alternatively, link to the live leaderboard instead of naming specific models.

**[embeddings.md] Contrastive loss temperature $\tau$ sensitivity** — Pedagogical reviewer: the temperature parameter in contrastive loss is mentioned but its sensitivity isn't — choosing the wrong $\tau$ can collapse training. Add a practical note if a dedicated contrastive learning page is written.

**[embeddings.md] FAISS index type tradeoffs** — Red Team: the concrete example uses `IndexFlatL2` (exact search) but doesn't mention that at scale you'd use approximate indexes (IVF, HNSW). Add a comment in the code or a note in the ANN search section. Defer to a dedicated vector search page.

**[embeddings.md] Negative sampling in word2vec** — Red Team: the skip-gram objective is shown correctly but the actual training procedure uses negative sampling (not the full softmax over the vocabulary). This distinction matters when implementing; add a parenthetical or defer to a word2vec deep-dive.

**[tokenization.md] BPE compression vs quality tradeoff** — Red Team: the merge count in the Mermaid trace is illustrative, not representative. Real BPE uses thousands of merges; the visual makes it look like a small fixed set. Add a note on scale if the tokenization page is expanded.

**[tokenization.md] ChatML format vs other instruction formats** — Pedagogical reviewer: the special tokens section mentions `<|im_start|>` / `<|im_end|>` (OpenAI ChatML) but Llama 3, Claude, and Gemini use different delimiters. This is correct but might imply ChatML is universal. Add a note that instruction formats are model-family-specific.

**[tokenization.md] Fertility metric for multilingual comparison** — Red Team: "tokens per word" is an English-centric metric. Non-European languages with morphologically complex structures (Turkish, Finnish) or logographic systems (CJK) tokenize very differently. The Thai example in the code illustrates this; connecting it to "fertility" (tokens/character) in the vocabulary table would be more rigorous.

**[training-vs-inference.md] Constitutional AI depth** — Pedagogical reviewer: Constitutional AI is described in one sentence. Given Anthropic's relevance to the target audience, a fuller explanation (self-critique loop, revision, principle set) would be valuable. Expand if a dedicated alignment page is written.

**[training-vs-inference.md] DPO vs PPO practical differences** — Red Team: DPO is presented as straightforwardly simpler than RLHF/PPO without noting that DPO can be sensitive to the reference model quality and doesn't allow online data collection. If a fine-tuning comparison page is written, add these tradeoffs.

**[training-vs-inference.md] Speculative decoding acceptance rate** — Red Team: speculative decoding is described without mentioning that the acceptance rate (fraction of draft tokens accepted) varies widely by domain and draft model quality. Add a sentence if the inference optimization section is expanded.

**[training-vs-inference.md] QLoRA PEFT library version sensitivity** — Red Team: the QLoRA code uses `BitsAndBytesConfig` which requires specific `bitsandbytes` versions and CUDA support; this breaks on MPS (Apple Silicon) and CPU-only machines. Add a hardware note or a `try/except` import if the example is expanded for broader audience.

**[training-vs-inference.md] "0.08% trainable parameters" is architecture-dependent** — Pedagogical reviewer: the 0.08% figure is for one specific LoRA rank and target modules; it will differ significantly with different rank or targeting all linear layers. Qualify as "approximately 0.08% with r=16 targeting q/v projections."

**[how-llms-work.md] System prompt confidentiality framing** — Red Team: the page notes that system prompts are "just text prepended to the conversation" but doesn't note that many providers instruct models to keep system prompts confidential. This creates a practical tension worth noting for API users. Add to the chat formatting section.

**[how-llms-work.md] "LLMs are compression artifacts" — accuracy of the metaphor** — Red Team: the My Take framing is poetic but technically imprecise — compression implies lossless or near-lossless; LLM generation is lossy and stochastic. The metaphor is useful but flag it as metaphorical ("in the sense that…") to avoid it being taken literally.

**[how-llms-work.md] Logit lens / mechanistic interpretability** — Pedagogical reviewer: the page covers the output of logit projection but doesn't mention that researchers use intermediate layer logit projections ("logit lens") to understand what the model is "thinking" at each layer. Defer to a mechanistic interpretability page.

**[how-llms-work.md] Beam search not mentioned** — Red Team: the sampling section covers temperature, top-p, top-k but omits beam search, which is still used in some translation and summarization systems. Add a row to the sampling table if the generation section is expanded.

**[how-llms-work.md] KV cache per-request vs per-token framing** — Pedagogical reviewer: the KV cache explanation is correct but describes it at the layer level; practitioners usually think about it per-request (total KV cache memory = batch size × seq length × layers × heads × head_dim × 2 × dtype). Add a total-memory formula if the serving section is expanded.

---

## Phase 3 review — 2026-04-23 (6 new pages: prompting, structured-outputs, function-calling, rag, multi-agent-systems, memory-architectures)

All must-fix and fix-in-current-pass items applied before phase close. Items below are deferred.

### Deferred — Pedagogical (add explanation / expand)

**[prompting.md] "Emerged" undefined for CoT** — Pedagogical: "CoT emerged spontaneously in sufficiently large models" — "emerged" means the model spontaneously started doing this without explicit training, but a new reader won't know this. Add: "meaning models of that scale began reasoning step-by-step when prompted, without being explicitly trained to do so."

**[prompting.md] Practical question checkpoints** — Pedagogical: "can you describe exactly what makes a good output different from a bad one?" is correct but vague. Add two concrete checkpoints: (1) Can you write a scoring rubric? (2) Can you write 3–5 good/bad output pairs?

**[prompting.md] Missing import in code example** — Red Team: the code example uses `anthropic.Anthropic()` without a visible `import anthropic` at the top. Add import for self-containedness.

**[prompting.md] Project ideas — missing benchmark references** — Pedagogical: "CoT ablation" project should reference the GSM8K dataset (available on HuggingFace) so readers have a concrete starting point.

**[structured-outputs.md] JSON mode constrained decoding vs retries — cost/latency difference** — Pedagogical: the distinction between token-level constrained decoding and retry-based enforcement is mentioned but the practical implication (cost, latency, reliability) isn't explained. Add: "constrained decoding blocks more tokens but guarantees valid JSON; retries are cheaper but can fail if recovery fails."

**[structured-outputs.md] Retry amplification — token cost** — Pedagogical: the failure mode says "30% of calls fail validation and retry succeeds = your schema is wrong" but doesn't make the cost implication explicit. Add: "This doubles your token cost for that operation."

**[structured-outputs.md] Project ideas — "parse failure" undefined** — Pedagogical: define the three categories to measure: (1) JSON parsing errors, (2) schema validation errors, (3) parse failures requiring manual intervention.

**[function-calling.md] Tool description good/bad pair** — Pedagogical: "description quality matters more than most people expect" is asserted but not shown. Add a concrete bad/good description pair for the same tool.

**[function-calling.md] Max iterations comment in loop** — Pedagogical: the `while True` execution loop has no guard. Add a comment: "# Safety: add a max_iterations check to prevent infinite loops — see Common failure modes."

**[function-calling.md] `tool_choice="none"` use case** — Pedagogical: the three tool_choice options are shown without explaining when to use "none". Add: "'none' is useful for the final response pass after all tool calls are done — prevents spurious tool calls when the model should just answer."

**[function-calling.md] KB search stub explanation** — Pedagogical: `search_knowledge_base` is a stub with a comment but a new reader doesn't know what "vector search" means. Add a one-line reference: "See the RAG page for how to build a vector search."

**[function-calling.md] Parallel call failure handling** — Red Team: `asyncio.gather()` doesn't show what happens if one task raises an exception. Add: "If one task fails, `gather()` raises the exception from the first failed task. Use `return_exceptions=True` to collect errors without cancelling other tasks."

**[rag.md] Pipeline diagram — query embedding step** — Pedagogical: the diagram shows "Query embedding" as a step but the text section explaining it comes later. Add one sentence before the diagram: "At query time, the same embedding model converts the user's query into a vector, which is matched against indexed chunks."

**[rag.md] Chunking size tuning guidance** — Pedagogical: "256–512 tokens is a reasonable default" without saying how to choose within that range. Add: "Test your corpus with both ends. If documents have many short distinct topics, use 256. If documents are narrative or have multi-sentence facts, use 512."

**[rag.md] Embedding model choice** — Pedagogical: `BAAI/bge-small-en-v1.5` is used without justification. Add: "BGE-small is a solid general-purpose default. For domain-specific corpora (medical, legal), look for domain-adapted variants. For corpora > 1M chunks, consider smaller embedders to reduce memory."

**[rag.md] Reranking latency — context for real-time** — Pedagogical: "~50–150ms" is stated without context. Add: "For real-time applications already waiting 100–300ms for retrieval, this is acceptable. If total latency budget is < 200ms, skip reranking or use a managed API (Cohere Rerank) that parallelizes better."

**[rag.md] Synthesis vs lookup — hierarchical retrieval path** — Pedagogical: "synthesis task → different approach" is stated without saying what approach. Add: "For synthesis, consider hierarchical retrieval: retrieve document summaries first, then full sections of the most relevant documents."

**[rag.md] Hallucination mitigation — concrete prompt and eval** — Pedagogical: the mitigation is "explicit prompt instruction" without showing what that looks like. Add a sample prompt and eval method: "Create 10 questions whose answers are NOT in your corpus; the model should say 'not found' rather than hallucinate."

**[multi-agent-systems.md] Validator agent — not shown in code** — Pedagogical: validator is mentioned in "Trust and verification" but no code or pseudocode is shown. Add pseudocode or reference the "Critic-revise loop" project idea explicitly.

**[multi-agent-systems.md] State update pattern** — Pedagogical: the TypedDict state is defined but it's not clear that agents return a dict with only the fields they update. Add: "# Each agent receives the full state and returns a dict containing only the fields it modifies."

**[multi-agent-systems.md] "Specialized handling" definition** — Pedagogical: "where each subtask benefits from specialized handling" is vague. Add: "Specialized handling means: different system prompt, different tools, different model tier, or different latency budget — not just a different variable in the same template."

**[multi-agent-systems.md] Project ideas — output quality metric** — Pedagogical: "compare output quality" in the three-topology project has no rubric. Add: "Score outputs on a 1–5 scale across accuracy, clarity, and actionability using a fourth Claude instance as judge."

**[memory-architectures.md] Context window size intuition** — Pedagogical: "doesn't fit in a single context window" is vague. Add: "200K tokens ≈ 150K words ≈ 300 pages. A 10,000-document knowledge base doesn't fit. 100 past conversations don't fit."

**[memory-architectures.md] In-context failure mode not shown** — Pedagogical: the code shows 4 turns without showing what happens at the boundary. Add a comment: "After ~100 turns (depending on length), older turns drop out of context. The model may forget early instructions. Mitigation: summarize or archive old turns."

**[memory-architectures.md] Conflicting stored memories example** — Pedagogical: "retrieval is approximate" is stated but an example of conflicting memories would make it concrete. Add: "If you store 'Prefer Python 3' and later 'Now using Python 2 for legacy project,' both may be retrieved — the model sees both and may default to the older one without recency scoring."

**[memory-architectures.md] Episodic memory — no usage example** — Pedagogical: `get_recent_events` is defined but never called in the section. Add: `recent = get_recent_events("user_123", limit=5); print([e['content'] for e in recent])`.

**[memory-architectures.md] Parametric memory — cost numbers** — Pedagogical: "expensive to update" without numbers. Add: "Fine-tuning a 7B model costs ~$100–1,000 in compute and must be rerun on each knowledge update. RAG costs ~$0.01–0.10 per query. Fine-tuning is worthwhile for stable knowledge queried thousands of times daily."

**[memory-architectures.md] Decision table — 20-turn boundary** — Pedagogical: "< 20 turns" lacks justification. Add footnote: "20 turns ≈ 4,000–8,000 tokens depending on message length; adjust based on your model's context window."

**[memory-architectures.md] Memory poisoning — no code for confidence field** — Pedagogical: the fix suggests adding a `confidence` field but the code example doesn't show it. Add pseudocode: `store_memory(user_id, content, source="user_stated", confidence=0.5)`.

**[memory-architectures.md] Vector DB filter testing** — Red Team: user_id filtering assumes the DB correctly isolates per-user, but this can fail under misconfiguration. Add: "Test namespace isolation explicitly: store memories for user_A, query as user_B, verify zero cross-user retrieval."

**[memory-architectures.md] Embedding model switching cost** — Red Team: switching models mid-deployment requires re-embedding the entire corpus — this cost is not warned about. Add: "Switching embedding models requires re-embedding all stored memories. Plan your model choice before indexing large corpora."

**[memory-architectures.md] Recency scoring — implementation path** — Red Team: "add recency scoring" is mentioned without specifying how. Add: "Two approaches: (a) filter to memories < 90 days old before semantic search, or (b) re-weight retrieved scores by age. Approach (a) is simpler; approach (b) allows older memories to surface when strongly relevant."

**[multi-agent-systems.md] Cost claim — retries not included** — Red Team: "$0.01/call × 3 = $0.03" assumes no retries. Add: "Add retries, error handling, and validation loops and actual cost is often 2–3× higher. Monitor per-agent token usage from day one."

**[rag.md] Embedding staleness warning** — Red Team: switching embedding models mid-deployment mixes incompatible vectors in the same index. The failure modes section mentions it but the code doesn't. Add to concrete example: "If you switch embedding models, delete and rebuild the index — mixing embeddings from different models produces nonsensical retrieval."

---

## Phase 4 Batch A review — 2026-04-23 (evals, observability, guardrails, prompt-injection, output-validation)

All 6 must-fix items applied before batch close. Items below are deferred.

### Deferred — evals.md

**[evals.md] Offline vs online eval distinction** — Pedagogical: doesn't distinguish offline evals (CI/CD regression detection) from online evals (continuous production monitoring). Add: "Evals run offline before deployment to catch regressions; monitoring-grade evals run continuously on production traffic to detect real-world drift."

**[evals.md] Drift detection coverage** — Pedagogical: "distribution shift" is listed as a problem but no concrete approach to detecting it is shown. Add a subsection: run your eval set weekly, track mean score over time, alert when score drops > N%.

**[evals.md] LLM judge calibration specifics** — Pedagogical: "validated judge-human agreement on a sample" doesn't specify what counts as sufficient. Add: "≥80% agreement rate on 50+ examples from your actual use cases before trusting the judge for automation."

**[evals.md] Eval set construction prioritization** — Pedagogical: four sources listed without priority for a team starting from scratch. Add: "Priority: (1) hand-author 20 edge cases you know are hard, (2) sample real interactions monthly once in production, (3) use public benchmarks only if your task closely matches."

**[evals.md] Failure mode 1 / My take redundancy** — Pedagogical: "Eval set memorization" failure mode (line 252) largely repeats the My take callout (lines 231–235). Either expand the failure mode to cover a distinct aspect or merge.

**[evals.md] Judge calibration not shown in concrete example** — Red Team: the concrete example doesn't validate that the judge is reliable before trusting its scores. Add a note: "Before using this judge in automated evals, calibrate on 50 examples with human labels. If agreement is below 80%, improve the judge prompt."

### Deferred — observability.md

**[observability.md] Multi-tenant trace correlation** — Pedagogical: single-request framing misses multi-user systems where you need to correlate logs per user. Add: "In multi-user systems, each request must carry user_id and session_id so logs from different users don't collide."

**[observability.md] PII redaction / debugging tension** — Pedagogical: redacting PII makes debugging harder; this tension isn't addressed. Add pattern: "Store redacted prompts in hot logs; store full prompts encrypted in cold storage with gated access for compliance/support investigations."

**[observability.md] Failure mode 2 (No session ID) consequence** — Pedagogical: states "every request looks independent" without showing what breaks. Add concrete break: "User reports 'the bot forgot my name,' but you see 10 separate traces with no link between them — impossible to debug."

**[observability.md] OTel code missing backend emit** — Pedagogical: the OpenTelemetry example doesn't connect to a backend. Add a simple exporter configuration (OTLP, Langfuse, etc.) so the snippet is production-usable.

**[observability.md] Staging table missing rationale** — Pedagogical: "Prototype → structured logging, Beta → Langfuse" without explaining why to wait. Add rationale for each tier transition.

### Deferred — guardrails.md

**[guardrails.md] Topic classifier false positive guidance** — Pedagogical: example doesn't include measurement of false positive rate or guidance on when to tune. Add: "Test on 100 real questions (including edge cases). If > 2% are blocked incorrectly, the classifier is too aggressive."

**[guardrails.md] Concrete example vs FP warning inconsistency** — Pedagogical: failure mode 2 warns against > 1–2% FP rate, but the concrete example doesn't include any FP measurement. Add: "After deploying, sample 20 blocked requests and manually verify whether they should have been blocked."

**[guardrails.md] Input vs output guard when-to-use** — Pedagogical: doesn't say when input guards suffice vs. when output guards are also needed. Add: simple chat → input guards; RAG bot → output faithfulness check; agent → output action classifier.

### Deferred — prompt-injection.md

**[prompt-injection.md] "What it is" / "Problem it solves" overlap** — Pedagogical: both sections make essentially the same point. Consider merging and moving specific consequence examples (customer support bot, agent exfiltration) to a "Why it matters" block.

**[prompt-injection.md] Defense failure modes per defense** — Pedagogical: each defense is listed without saying what it fails against specifically. Add a defense comparison table: position separation fails against role-play; output monitoring fails if goal isn't exfiltration; etc.

**[prompt-injection.md] Indirect injection frequency context** — Pedagogical: examples list indirect injection surfaces without noting which are most common in practice. Add: "RAG document injection is the most common real-world vector. Code comment injection is less common (fewer teams have agents that read arbitrary code)."

**[prompt-injection.md] Stored injection in concrete example** — Pedagogical: failure mode 7 mentions sanitizing stored DB records, but the concrete example only sanitizes retrieved docs. Extend example to show: retrieve docs from DB, check against injection patterns before using.

### Deferred — output-validation.md

**[output-validation.md] instructor retry transparency** — Pedagogical: "schema enforcement + retry included" without showing what the retry looks like. Add: "If the first attempt returns invalid JSON, instructor re-prompts with the validation error. This recovers ~80% of initial failures."

**[output-validation.md] Retry count guidance specifics** — Pedagogical: the My take callout says "have a ceiling" without specifying what count is appropriate or why. Add: "2–3 retries is typically the right ceiling. Beyond that, the problem is systematic — fix retrieval or schema, not retry count."

**[output-validation.md] Task-specific validation coverage** — Pedagogical: validation is shown for RAG Q&A but not for other common tasks (summarization, code generation). Add a "Validation for different tasks" subsection showing: summarization (does summary match source on key facts?) and code (syntax + test passage).

**[output-validation.md] "High-accuracy outputs" circular reasoning** — Pedagogical: "skip if model already produces high-accuracy outputs" is circular — you can't know this without validation. Reframe: "Skip if the user reviews output before acting, or if benchmarked validation overhead exceeds latency budget."

**[output-validation.md] False confidence from layered defense** — Red Team: layered defense (guardrails + validation) implies high reliability but doesn't discuss what adversarial inputs could still slip through. Add: "Layered defenses raise the bar but don't eliminate risk. Measure false negative rate of your validator, not just false positive rate."

---

## Phase 4 Batch B review — 2026-04-23 (vector-databases, caching, latency-optimization, model-routing, cost-tracking)

All 6 must-fix items applied before batch close. Items below are deferred.

### Deferred — vector-databases.md

**[vector-databases.md] Namespace isolation — embedding space collisions** — Red Team: metadata filtering prevents cross-user retrieval but doesn't prevent semantically similar documents from different users appearing near each other in the shared embedding space. Add: "Test three isolation scenarios: (1) metadata filtering prevents cross-user retrieval, (2) semantic cache (if enabled) does not leak personalized results, (3) semantically similar documents from different users don't surface through filter edge cases."

### Deferred — caching.md

**[caching.md] User isolation missing from semantic cache code** — Red Team: the semantic cache example doesn't include user_id in the cache key, making it unsafe for personalized responses in multi-user systems. Add a note in the code: "In multi-user systems, scope your cache by user_id — include it in the lookup key — to prevent returning user A's cached personalized response to user B."

**[caching.md] Cache poisoning** — Red Team: the stale cache failure mode doesn't mention that adversarial users can inject misleading responses into a shared semantic cache. Add: "In multi-tenant systems, semantic caches can be poisoned — a malicious user caches a false response, which then returns to other users with similar queries. Mitigate by isolating caches per trust boundary."

### Deferred — model-routing.md

**[model-routing.md] Routing heuristic misclassification rate** — Red Team: the regex patterns will misclassify 20–30% of real queries. The My take already recommends testing against 100 real queries; add: "If > 25% are misclassified, use an LLM router instead — even with latency overhead, the quality improvement is usually worth it."

**[model-routing.md] Fallback chain — error type distinction** — Red Team: the fallback chain catches all exceptions equally, including rate limits (should retry same model) and 400-level errors (should fail immediately). Add: "Distinguish error types: retry on rate limits/timeouts (same model with backoff), fall back on 503/unavailable, fail immediately on 400-level errors."

**[model-routing.md] LLM router break-even analysis** — Red Team: the LLM-as-router example doesn't analyze when the router cost/latency overhead is worth it. Add: "LLM routers add ~100–150ms and their own token cost. Break-even: the cost savings from better routing must exceed the router's cost per query. Measure both before committing to an LLM router."

**[model-routing.md] "medium_reasoning" → Haiku mapping** — Technical: routing medium-complexity queries to Haiku may surprise users expecting Sonnet-level quality. Add a note: "This mapping routes medium queries to Haiku — verify with eval that quality is acceptable before deploying."

### Deferred — cost-tracking.md

**[cost-tracking.md] Pricing staleness in concrete example** — Red Team: the second code block (lines 225–230) uses scientific notation pricing without the warning from the first block about keeping pricing in config. Unify both examples to load pricing from config or at least duplicate the warning.

**[cost-tracking.md] Failed call cost tracking** — Red Team: code examples don't track failed-call token spend. Add: "Wrap all API calls in try/except and track failed calls separately. At a 5% error rate, tracked cost is only 95% of actual spend."

**[cost-tracking.md] Conversation cost analysis heuristic accuracy** — Red Team: `len(t["content"].split()) * 1.3` is an approximation that's off by 20–30% for special characters, code, and non-English content. Add: "For accurate cost tracking, use the actual input_tokens from each API response rather than word-count heuristics."

---

## Phase 4 Batch D review — 2026-04-23 (knowledge-graphs, red-teaming, audit-logs, access-control)

All must-fix items applied before batch close. Items below are deferred.

### Deferred — knowledge-graphs.md

**[knowledge-graphs.md] MERGE stores relationship type as property, not as Cypher type** — Technical: `MERGE (s)-[r:RELATION {type: $predicate}]->(o)` encodes semantics as a property on a generic RELATION type. This defeats index-backed relationship traversal and makes access control by relationship type impossible. Note in the Neo4j section: "This schema design stores all edge semantics as properties on a single RELATION type. Production graphs typically use distinct Cypher relationship types (e.g., `[:REPORTS_TO]`, `[:WORKS_AT]`) for better query performance and access control granularity."

**[knowledge-graphs.md] multi_hop() includes source entity** — Minor: `single_source_shortest_path_length` with `cutoff=hops` returns distance 0 for the source node, so it's included in the returned set. Functionally harmless but potentially surprising. Add a note: "The source entity is included at distance 0. Filter with `if v > 0` in the dict comprehension if self-exclusion is needed."

**[knowledge-graphs.md] Cypher syntax orientation** — Pedagogical: Cypher `(node)` / `-[edge]->` pattern not explained for readers unfamiliar with it. Note added to "Neo4j for production scale" header covers this partially; expand if a dedicated graph DB page is added.

**[knowledge-graphs.md] ALLOWED_PREDICATES in concrete example** — The concrete example `build_graph_from_docs` doesn't apply the ALLOWED_PREDICATES filter from `extract_triples`. Unify: either use `extract_triples` in the concrete example or duplicate the validation there.

### Deferred — red-teaming.md

**[red-teaming.md] Manual red team scope and team size** — Pedagogical: not clear whether this is a solo activity or requires a team, or whether it's sequential vs. iterative. Clarify: "typically 2–4 people working independently to avoid shared blind spots, with findings collated and retested after mitigations."

**[red-teaming.md] Attacker model capability requirement** — If attacker model is weaker than target, automated red team will have systematically low coverage. Add: "Attacker model capability must meet or exceed target capability for results to be meaningful. Low attack success rates from a weaker attacker are not evidence of safety."

**[red-teaming.md] "Authority pretense" name ambiguity** — Pedagogical: "authority" is ambiguous. Clarify: this means prompts that claim the user has special permissions over the model's constraints — not external legal or governmental authority.

### Deferred — audit-logs.md

**[audit-logs.md] Concrete example writes to local disk (non-compliant for regulated workloads)** — The concrete example has a note in the caution block about compliance; the code itself still shows `Path("logs/audit.jsonl")` without further warning. Reinforce with an inline comment: `# Production: replace with a write-once sink (S3 Object Lock, CloudWatch Logs, managed SIEM)`.

**[audit-logs.md] Credit card regex misses Amex (15 digits)** — The `\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}` pattern matches 16-digit cards only. Amex is 15 digits (4-6-5 grouping). Add Amex pattern or expand the caveat that regex-based PAN detection is not PCI-DSS sufficient.

**[audit-logs.md] cold tier duration_days comment** — The comment "example only — determine from legal/compliance requirements" is correct; reinforce in the nearby prose that 7 years is neither a universal minimum nor maximum.

### Deferred — access-control.md

**[access-control.md] run() loop for-loop vs while-loop** — The iteration limit changes `while True` to `for _ in range(max_iterations)`, but the concrete example's `run_support_agent` function still uses `while True` without a limit. Apply the same pattern there for consistency.

**[access-control.md] register_tool schema population — no example** — The comment notes properties must be populated but doesn't show an example with a real schema. Add a documented example for `read_document`: `"properties": {"path": {"type": "string", "description": "Absolute path to the file to read"}}, "required": ["path"]`.

**[access-control.md] Transitive permission escalation** — Red Team noted that a tool that calls another service with broader permissions is an escalation path not detected by the registry. Add to failure modes: "Audit the full permission chain of each tool — a tool that triggers a downstream service with admin credentials escalates beyond its declared scope regardless of what the registry says."

---

## Phase 4 Batch C review — 2026-04-23 (orchestration-frameworks, mcp, model-distillation, synthetic-data, document-processing)

All 5 must-fix items applied before batch close. Items below are deferred.

### Deferred — orchestration-frameworks.md

**[orchestration-frameworks.md] "Start with the raw SDK" — no signal for when to switch** — Pedagogical: "start with the raw SDK" is sound advice but the page gives no criteria for when a framework becomes worth its overhead. Add: "Reach for a framework when you need integrations it already has (document loaders, vector store connectors, pre-built agent loops) and the integration would take a day to write from scratch."

**[orchestration-frameworks.md] DSPy optimizer eval consumption** — Red Team: "runs DSPy's optimizer on a 50-example eval set" in the project idea doesn't warn that the optimizer makes many LLM calls against the eval set. Add: "DSPy's optimizer makes many internal LLM calls; budget ~5–20× your eval set size in total API calls when running optimization."

**[orchestration-frameworks.md] Framework version pinning — no example** — Pedagogical: "pin your framework version in requirements.txt" is stated without showing what safe pinning looks like. Add: `langchain-anthropic==0.3.x` rather than `langchain-anthropic>=0.3` — distinguish minor-version pinning from exact pinning.

**[orchestration-frameworks.md] LangGraph vs function composition clarity** — Pedagogical: the StateGraph example shows graph compilation but doesn't explain why you'd use a graph vs a sequence of function calls. Add: "Use LangGraph when you need conditional edges (branch on state), cycles (retry nodes), or built-in checkpointing. For linear pipelines, direct function composition is simpler."

### Deferred — mcp.md

**[mcp.md] Tool name collision across servers** — Pedagogical: the multi-server example doesn't address what happens if two MCP servers expose tools with the same name. Add: "If two servers expose a tool with identical names, the host may silently shadow one. Prefix tool names with a namespace (e.g., `filesystem__read_file`, `db__read_file`) in your server implementations."

**[mcp.md] Resources and Prompts capability underexplained** — Pedagogical: Resources and Prompts are listed as server capability types but not shown in code or examples. The page focuses entirely on Tools. Add a sentence: "Resources and Prompts are rarely needed in custom servers; most real-world MCP servers expose Tools only. See the MCP spec if you need to expose data sources or reusable prompt templates."

**[mcp.md] HTTP+SSE authentication gap** — Red Team: the transport section notes HTTP transport requires authentication but doesn't say what's standard. Add: "For HTTP transport, use Bearer token authentication. The stdio trust model (inherited OS permissions) does not apply."

### Deferred — model-distillation.md

**[model-distillation.md] OOD detector — no code** — Pedagogical: the My take recommends adding an OOD routing layer but provides no implementation guidance. Add a pointer: "See Project idea #3 for an embedding-similarity OOD detector implementation."

**[model-distillation.md] LoRA target modules completeness** — Red Team: the LoRA config targets only `q_proj` and `v_proj` but many practitioners also target `k_proj`, `o_proj`, and the FFN layers for better adaptation. Add a comment: "# Targeting q/v is the minimum; for deeper adaptation add k_proj, o_proj, gate_proj. More targets → better fit, slower training, slightly more parameters."

**[model-distillation.md] Temperature T² scaling intuition** — Pedagogical: the T² scaling factor is shown in the loss but not explained. Add: "The T² factor compensates for the fact that soft targets at temperature T have T²-times-smaller gradients than hard targets. Without it, the distillation term would be underweighted as temperature increases."

### Deferred — synthetic-data.md

**[synthetic-data.md] Deduplication order-dependence bias** — Red Team: the greedy deduplication algorithm (keep the first occurrence, drop near-duplicates) introduces order-dependence bias — examples appearing earlier in the list are always kept over later ones regardless of quality. Add: "Sort examples by quality score before deduplication so higher-quality examples are kept over lower-quality ones when both would be deduplicated."

**[synthetic-data.md] Category balance check — fix path missing** — Pedagogical: the concrete example prints the distribution imbalance but doesn't show how to fix it. Add: "If one category has < 80% of the target count, re-run `generate_classification_data` for that category only and append to the dataset."

**[synthetic-data.md] Real data mixing ratio — no calibration path** — Pedagogical: "even 10–20% real examples in the training mix helps" (My take) gives a ratio without explaining how to calibrate it for your task. Add: "Run a small ablation: train at 0%, 10%, 20%, 30% real data on a held-out real test set. The quality curve typically flattens after 20%."

### Deferred — document-processing.md

**[document-processing.md] OCR 70% text-coverage threshold — arbitrary** — Red Team: `if result["text_coverage"] >= 0.7` is an arbitrary threshold without calibration guidance. Add: "The 70% threshold works for most mixed documents. For documents where even 10% image-only pages matter (legal exhibits, engineering drawings), lower to 0.3 and route partial-text PDFs to hybrid extraction."

**[document-processing.md] Vision-LLM extraction not mentioned** — Pedagogical: the page covers text extraction + OCR but omits using a vision LLM (Claude's vision API, GPT-4o) to process PDF pages as images — a valid approach for complex layouts where neither text extraction nor OCR work well. Add a note in the tools table: "Claude vision API / GPT-4o vision: page-as-image extraction; handles complex multi-column and diagram-heavy layouts that defeat text extractors. Higher cost; reserve for documents where layout fidelity is critical."

**[document-processing.md] Repeated header detection — no implementation shown** — Pedagogical: "detect and strip repeated patterns across pages" is listed as a failure mode fix without any code. Add pseudocode: collect the first line of each page's extracted text; if a line appears in > 50% of pages, flag it as a header and strip it before chunking.

**[document-processing.md] Chunk overlap — not implemented in concrete example** — Red Team: the chunk overlap (`overlap = 40`) is defined but the loop logic (`range(0, len(words), chunk_size - overlap)`) sets stride = 310 words, meaning each chunk starts 310 words after the previous — correct, but the last partial chunk isn't guarded. Add: `if len(chunk_words) < chunk_size // 2: break` to skip orphan chunks shorter than half the target size.

---

## Phase 5 review — 2026-04-23 (reliability, graceful-degradation, confidence-estimation, fallbacks, monitoring, deployment-patterns)

All 19 must-fix items applied before phase close. Items below are deferred.

### Deferred — reliability.md

**[reliability.md] Timeout fix not actionable** — Pedagogical: "set timeouts based on expected completion length, or use streaming with per-chunk timeouts" is directionally correct but gives no implementation guidance. Add: a practical rule of thumb (e.g., `timeout = max_tokens / 10` seconds as a starting estimate) and a pointer to the Anthropic SDK streaming docs for per-chunk timeout patterns.

**[reliability.md] Output validation underemphasized in code** — Pedagogical: the circuit breaker dominates the example but silent truncation (finish_reason == "max_tokens") is the more universally relevant failure. Consider a secondary code snippet that focuses solely on output validation without the circuit breaker overhead.

### Deferred — graceful-degradation.md

**[graceful-degradation.md] Feature-level degradation not shown in code** — Pedagogical: the concept ("disable only the AI-powered feature, not the entire application") is explained but no code or pseudocode shows what this looks like. Add a stub showing an AI feature as an optional enrichment layer that can be bypassed when `ai_enabled=False`.

**[graceful-degradation.md] Mechanism of degradation masking regressions** — Pedagogical: "alert when degradation rate spikes" is the correct fix, but the failure mode doesn't explain why standard monitoring misses it — because the HTTP layer sees successful responses (200 OK), and the quality layer evaluates fewer primary-model responses, making the regression invisible without explicit degradation-rate tracking.

**[graceful-degradation.md] Threshold calibration fix is not actionable** — Pedagogical: "tune thresholds against held-out labeled data" is correct but vague. The "Threshold calibration tool" project idea shows the cost-minimization approach — link to it from the failure mode entry so the reader knows where to look.

### Deferred — confidence-estimation.md

**[confidence-estimation.md] Self-consistency API transmission multiplier unremarked** — Security (S9 deferred): each of N samples is a separate API transmission of the user's input. In regulated environments (HIPAA, GDPR), repeated transmission may be subject to DPA constraints. Add: "Each sample is a separate API transmission of the user's input; in regulated environments, verify that repeated transmission is permitted under your data processing agreement before using self-consistency at high N."

**[confidence-estimation.md] Verbalized confidence adversarially inflatable** — Security (S10 deferred): the existing caveat notes sycophancy but not adversarial manipulation. A user who knows the system routes on verbalized confidence can include affirmation cues ("you are very certain about this") to inflate the score. Add: "Verbalized confidence is also adversarially inflatable — prompts containing affirmation cues cause models to self-rate higher. Never use it as a routing gate in public-facing or adversarial contexts."

**[confidence-estimation.md] Self-consistency cost not in example narrative** — Pedagogical: the 10x cost implication appears only in "When to use it" but the concrete example runs 10 samples without any cost framing in the example narrative itself. Add a comment in the example showing the cost calculation: `# 10 samples = 10x API cost; at $3/Mtok, 1000 queries/day at 500 tokens = $15/day just for confidence estimation`.

### Deferred — fallbacks.md

**[fallbacks.md] Streaming stitching risk not explained** — Pedagogical: "restarting is simpler and safer" is stated without explaining why stitching is dangerous. The risk: the fallback model hasn't seen the partial response in its context, so stitching produces incoherent continuations. Add this mechanism so readers understand when the complexity of stitching might be worth attempting.

**[fallbacks.md] "Test fallback behavior explicitly" is underspecified** — Pedagogical: the system prompt differences failure mode recommends testing but doesn't say how. Add: "Run your golden dataset eval against each model in the fallback chain separately, not just the primary. Divergences > 10% in task-specific metrics (format compliance, answer quality) indicate the fallback needs a separate system prompt."

### Deferred — monitoring.md

**[monitoring.md] Input distribution shift code example missing** — Pedagogical: input distribution monitoring is identified as one of four core techniques but isn't shown in the code example. The example covers LLM-as-judge, structural metrics, and latency — but not embedding-based drift detection. Add a second code snippet showing how to compute cosine distance from a baseline embedding centroid.

**[monitoring.md] Same-model judge failure mechanism not explained** — Pedagogical: "the judge shares the same failure modes" is stated but not explained. The underlying reason: a model that systematically hallucinations in a domain will also systematically rate hallucinations in that domain as correct, because the error is correlated, not random. Without this mechanism, "use a different model" reads as an arbitrary preference.

### Deferred — deployment-patterns.md

**[deployment-patterns.md] A/B outcome metrics for LLMs unspecified** — Pedagogical: the page explains the statistical requirements (stopping rule, significance level) but not what to measure. Add: "For chat applications, prefer thumbs-up rate or task completion rate over LLM judge scores as the primary A/B metric — judge scores can be gamed by prompts that sound good to the judge but not to users."

**[deployment-patterns.md] Shadow mode cost not addressed** — Pedagogical: shadow mode doubles inference cost (two full model calls per user request). The example doesn't address this. Add to the "When to use" section: "Shadow mode costs 2× inference for the duration it runs. Run it for 1–3 days on a sampled subset of traffic (10–20%) rather than all production traffic to limit cost."

---

## Final site-wide review — 2026-04-23 (all 56 pages)

Three review agents (Technical Accuracy, Pedagogical, Security) + Editor/Format agent. 23 items applied before this note was written. Items below are deferred.

### Deferred — Security (advisory)

**[rag.md] SSRF for URL-based retrieval** — Security (advisory): the RAG example uses a vector index, not live URL fetching. However, real-world RAG systems often fetch content from URLs or external APIs. Add a note: "If your retrieval layer fetches content from URLs at query time, validate those URLs before fetching — adversarial document metadata can inject SSRF targets."

**[multi-agent-systems.md] No error handling between agent outputs** — Security (advisory): the concrete example passes raw text from one agent to the next without checking quality or catching hallucinations. Add a comment: "Production agent pipelines should validate each agent's output before passing it downstream — consider a validator agent or output schema check between stages."

**[guardrails.md] Regex-first detection overemphasized** — Security (advisory): the concrete example leads with regex blocklists despite the page acknowledging that sophisticated attacks use paraphrasing and role-play. The existing LLM classifier section already covers the better approach; consider reordering so LLM classification is presented before regex as the primary recommended layer.

### Deferred — Pedagogical

**[intro.md] Graceful Degradation vs Fallbacks distinction not explained** — Pedagogical: the production concerns reading path lists both concepts without clarifying the relationship. Graceful Degradation is about reducing capability level; Fallbacks is about switching to a different model. One sentence distinguishing them would help readers prioritize.

**[glossary.md] Calibration definition is dense for a first read** — Pedagogical: the definition dives into ECE before making "stated confidence matches actual accuracy" concrete. Add a one-sentence worked example: "A model that says '70% confident' on 100 questions should get ~70 of them right."

**[glossary.md] RSSM entry implies prior knowledge of world models** — Pedagogical: the RSSM glossary entry references "Dreamer-style model-based RL" which itself needs context. Add: "An RSSM is what makes Dreamer-style RL practical — it lets the agent plan in imagination without environment calls."

**[glossary.md] MCP entry too high-level** — Pedagogical: "Anthropic's open protocol for connecting LLMs to external tools and data sources" doesn't explain the problem it solves. Add: "Without MCP, every tool integration is bespoke — MCP standardizes the interface so any MCP-compatible tool works with any MCP-compatible host."

**[attention.md] √d_k scaling intuition incomplete** — Pedagogical: the math is correct but the connection between large dot product variance → softmax saturation → vanishing gradients is left implicit. Add one sentence: "Without the scaling, attention collapses to near-one-hot distributions in deep layers, causing the gradient of the softmax to vanish and training to stall."

**[attention.md] Head specialization vs "attention ≠ explanation" tension** — Pedagogical: the page claims heads learn distinct relationships (syntax, coreference) AND warns that attention ≠ explanation. These are in tension and neither is resolved. Add a reconciling note: "Head specialization is real at the coarse level; the warning is about interpreting individual attention weights as causal explanations — they correlate with behavior but don't determine it."

**[tokenization.md] Thai tokenization claim needs grounding** — Pedagogical: "Thai requires more tokens per character" stated without explaining why (no whitespace boundaries, BPE trained on English-heavy corpus). Add one sentence on the mechanism.

**[prompt-injection.md] Position separation defense — WHY not explained** — Pedagogical: the page lists position separation as a defense without explaining why it works (or fails). Add: "Position separation exploits the model's tendency to treat earlier context as more authoritative, but it's a behavioral heuristic, not an architectural barrier — the model processes all positions with the same attention mechanism and can be prompted to override the separation."

**[observability.md] No reading path connection to evals** — Pedagogical: observability data (traces with prompts/outputs) is the natural input to eval construction, but the two pages don't cross-reference each other. Add a sentence at the end of the observability concrete example: "The traces you capture here are the raw material for your eval set — see [[Evals]] for how to turn production traces into regression tests."

**[vector-databases.md] Normalization warning lacks actionable fix** — Pedagogical: "Do not assume your vector database normalizes automatically" correctly warns the reader but doesn't say what to do. Add: "Fix: normalize vectors before inserting (`v / np.linalg.norm(v)`) or switch to cosine distance as the index metric — most databases support this as an index config option."

**[long-horizon-agents.md] Checkpoint resume behavior underexplained** — Pedagogical: the code shows saving/loading AgentState but doesn't explain what "resume" means from the agent's perspective. Add: "On resume, the agent reloads the checkpoint and continues from the next step in the plan — it doesn't re-run completed steps, and the context summary replaces the raw turn history from before the crash."

**[mcp.md] "Full permissions" warning lacks practical guidance** — Pedagogical: the warning correctly notes MCP servers inherit host permissions but doesn't say how to evaluate a third-party server. Add: "Before installing a third-party MCP server, check: what filesystem paths does it access? Does it make outbound network calls? Does the server have a published audit? Treat it like a VS Code extension — useful but not zero-trust."

**[memory-architectures.md] Mermaid diagram temporal sequence unclear** — Pedagogical: the flowchart shows `overflow → summarize into` but doesn't indicate what triggers the overflow check or when summarization fires. Add a label or note: "Summarization is triggered when token count exceeds a threshold (e.g., 80% of context window); the summary replaces raw turns, not the other way around."

**[how-llms-work.md] KV cache memory growth lacks mitigation** — Pedagogical: the page explains that KV cache grows linearly with context length but doesn't tell practitioners what to do about it. Add: "Mitigations: use vLLM's PagedAttention for production serving, set context limits per request, or use a streaming architecture that clears context between sessions."

## Phase 6 review — 2026-04-23 (agentic-computer-use, long-horizon-agents, world-models, multimodal-frontier, on-device-ai, novel-interaction, ai-safety)

Three review agents (Technical Accuracy, Pedagogical, Security) completed. 34 fixes applied before phase close. Items below were flagged but deferred.

**Note on recovery:** The specific text of 24 deferred items (4 security, 20 pedagogical — numbered S7, S8, S12, S16 and P1–P4, P7, P9–P12, P14–P16, P19–P20, P23–P28 in the review session) was not captured before context compression. To recover the full deferred list, rerun the three review agents against the current Phase 6 files. The 34 applied fixes are listed in the session summary.

### Known deferred — Security (4 items: S7, S8, S12, S16)

Locations inferred from numbering gaps in the approved fix list:

- **S7** — flagged in confidence-estimation.md or fallbacks.md (between S6 and S9 in the sequence). Content not recovered.
- **S8** — flagged in confidence-estimation.md or fallbacks.md. Content not recovered.
- **S12** — flagged in on-device-ai.md (between S11 and S13). Content not recovered.
- **S16** — flagged in novel-interaction.md or ai-safety.md (between S15 and S17). Content not recovered.

Action: rerun the Security review agent against Phase 6 files to recover these items.

### Known deferred — Pedagogical (20 items: P1–P4, P7, P9–P16, P19–P20, P23–P28)

Approved pedagogical fixes were P5, P6, P8, P13, P17, P18, P21, P22, P29. All other items in the P1–P29 range were deferred. Content not recovered from this session.

Action: rerun the Pedagogical review agent against Phase 6 files to recover the full deferred list.
