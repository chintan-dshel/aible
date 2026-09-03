---
sidebar_position: 3
title: History — Timeline
description: Key AI milestones from 1950 to the present, dense and scannable.
---

# History — Timeline

A reference timeline: what happened, when, and why it mattered. For the story connecting these events, see [History — Narrative](history-narrative.md).

---

## 1950s–1960s: The founding era

**1950** — Alan Turing, "Computing Machinery and Intelligence" — Poses the question "Can machines think?" and proposes the Imitation Game as a practical test. Establishes the conceptual frame that AI research would argue with for the next seventy years.

**1956** — Dartmouth Conference — John McCarthy, Marvin Minsky, Claude Shannon, and Nathaniel Rochester organize the summer workshop that coins the term "Artificial Intelligence." The field is born as a named discipline.

**1957** — Perceptron — Frank Rosenblatt builds the Mark I Perceptron at Cornell, a physical machine learning to recognize patterns from punch cards. The first practical implementation of a learned classifier.

**1958** — LISP — John McCarthy develops LISP at MIT, the programming language that would dominate AI research for the next thirty years. Symbolic computation as the substrate for intelligence.

**1965** — DENDRAL — Edward Feigenbaum and Joshua Lederberg at Stanford build the first expert system: DENDRAL infers molecular structure from mass spectrometer data. Demonstrates that encoded domain knowledge can perform at expert level.

**1966** — ELIZA — Joseph Weizenbaum at MIT creates ELIZA, a chatbot that mirrors user statements to simulate a psychotherapist. Users report feeling genuinely understood. Weizenbaum is disturbed by this; his 1976 book *Computer Power and Human Reason* is the earliest serious critique of anthropomorphizing AI.

**1969** — Minsky & Papert, *Perceptrons* — Prove that single-layer perceptrons cannot learn XOR (a pattern that can't be separated by a single straight dividing line, however you draw it) or any other non-linearly separable function. Widely (and somewhat unfairly) interpreted as refuting neural networks entirely. Contributes to the first AI winter.

---

## 1970s–1980s: Expert systems and first winter

**1972** — PROLOG — Alain Colmerauer develops PROLOG, a logic programming language. With LISP, forms the two dominant AI languages of the era.

**1974–1980** — First AI Winter — Funding cuts in the US and UK following failed promises. The Lighthill Report (1973) in the UK concludes that AI has failed to achieve its goals. DARPA ends its speech understanding program. Research continues but at reduced scale.

**1980** — XCON (R1) — Digital Equipment Corporation deploys XCON, an expert system that configures VAX computer orders. Reportedly saves DEC millions of dollars per year in configuration errors (the often-cited $40M annual figure is poorly sourced and likely cumulative). Triggers massive corporate investment in expert systems through the mid-1980s.

**1981** — Japan's Fifth Generation Computer Project — Japan announces a ten-year, $850M program to build AI-capable hardware and software. Triggers matching investment in the US and Europe. Much of the promised capability is not delivered.

**1986** — Backpropagation popularized — Rumelhart, Hinton & Williams publish "Learning representations by back-propagating errors" in *Nature*. Backpropagation is the algorithm that lets a network learn from its mistakes: run an example through, compare the output to the right answer, then work backward through the network adjusting each connection by how much it contributed to the error. It was independently discovered earlier (Werbos, 1974; LeCun, 1985) but this paper's clarity and framing re-ignites interest in neural networks.

**1987–1993** — Second AI Winter — The expert systems market collapses as maintenance costs balloon and limitations become clear. LISP machine companies fail. The AI industry loses most of its commercial funding for several years.

---

## Late 1980s–1990s: Connectionism and statistical learning

**1989** — Convolutional networks — Yann LeCun at Bell Labs applies convolutional neural networks (a network design that scans small patches of an image with the same learned filter, well suited to recognizing shapes wherever they appear) to handwritten digit recognition. LeNet achieves state-of-the-art on the USPS zip code dataset. The ideas are correct; the hardware is not yet adequate.

**1997** — LSTM published — Hochreiter & Schmidhuber, "Long Short-Term Memory" (*Neural Computation*). The core ideas were developed in Hochreiter's 1991 diploma thesis but went unnoticed for years; the 1997 paper is when the field adopted LSTMs. Recurrent networks (networks that process a sequence step by step, feeding each step's output back in as input to the next) had a problem where the learning signal shrank to nothing over long sequences, so early parts of a long input stopped influencing what the network learned — LSTM solves that "vanishing gradient" problem, enabling sequence modeling over long time horizons.

**1995** — Support Vector Machines — Corinna Cortes and Vladimir Vapnik publish the SVM paper. SVMs (a method that finds the best dividing boundary between categories, then a variant called kernel methods that lets that boundary bend to fit non-straight patterns) dominate machine learning benchmarks through the 2000s, outperforming neural networks on most tasks with less compute. The theoretical foundations (VC dimension and structural risk minimization — mathematical tools for predicting how well a method will generalize to new data before you've even tested it) give the field a rigorous grounding.

**1997** — Deep Blue beats Kasparov — IBM's Deep Blue defeats world chess champion Garry Kasparov 3.5–2.5 in a six-game match. The win is based on search and evaluation functions, not learning. The AI community is ambivalent: chess was supposed to prove intelligence; Deep Blue proves it is "just search."

**1998** — LeNet-5 — Yann LeCun publishes the definitive convolutional network paper and the MNIST benchmark. The MNIST dataset becomes the standard benchmark for handwritten digit recognition for the next two decades.

---

## 2000s: The shallow ML era

**2001** — Random Forests — Leo Breiman publishes random forests. Ensemble methods — combining many weak learners — become the dominant approach for tabular classification tasks.

**2001** — Wikipedia launches — Within a decade, Wikipedia provides a massive corpus of structured human knowledge in machine-readable form. Its influence on language model pretraining is enormous.

**2006** — Hinton's deep belief networks — Geoffrey Hinton, Simon Osindero, and Yee-Whye Teh show that deep networks can be pre-trained layer-by-layer using restricted Boltzmann machines — an earlier kind of network trained to reconstruct its own input, used here one layer at a time as a warm-start before the whole network is fine-tuned together. Reignites interest in deep learning after a decade of SVM dominance.

**2009** — ImageNet — Fei-Fei Li and colleagues at Stanford release ImageNet: 14 million labeled images across 20,000 categories. The dataset that would make the deep learning revolution possible.

**2009** — XGBoost precursors — Gradient boosted trees (Friedman, 2001) — an ensemble method that builds many small decision trees in sequence, each one correcting the errors of the ones before it — are refined throughout the decade. XGBoost is published by Chen & Guestrin in 2016, but the gradient boosting family dominates tabular ML competitions through this period.

---

## 2010s: The deep learning revolution

**2012** — AlexNet — Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton win the ImageNet Large Scale Visual Recognition Challenge with a 15.3% top-5 error rate, compared to the second place's 26.2%. The gap is large enough that the field reroutes. GPUs, large datasets, and deep convolutional networks become the dominant paradigm overnight.

**2013** — word2vec — Tomas Mikolov and colleagues at Google publish word2vec: it represents each word as a list of numbers (a "vector"), trained by learning which words tend to appear near which other words in real text ("co-occurrence statistics"). For the first time, words have geometric relationships — "king − man + woman ≈ queen." Transfer learning (reusing something learned on one task as a head start on another) begins for NLP.

**2014** — GANs — Ian Goodfellow and colleagues introduce Generative Adversarial Networks. A generator network and a discriminator network compete; the generator learns to produce realistic samples. Launches a decade of generative modeling research.

**2014** — Attention (seq2seq) — Bahdanau, Cho & Bengio publish the first attention mechanism for neural machine translation: instead of a network reading the whole input sentence ("the encoder") and squeezing everything it learned into one fixed-size summary vector before the output-generating side ("the decoder") ever starts writing, attention lets the decoder look back at every part of the original sentence directly, each time it produces a new word.

**2015** — ResNets — He, Zhang, Ren & Sun at Microsoft Research publish Deep Residual Networks, enabling training of 152-layer networks via residual connections. Win the 2015 ImageNet competition by a large margin. Residual connections become standard across virtually every deep architecture.

**2016** — AlphaGo — DeepMind's AlphaGo defeats world Go champion Lee Sedol 4–1. Unlike Deep Blue, AlphaGo learns from human games and self-play using deep reinforcement learning. Go was supposed to be too complex for brute-force search; AlphaGo renders the argument moot.

**2017** — "Attention Is All You Need" — Vaswani et al. at Google publish the Transformer architecture: attention mechanisms replace recurrence (the step-by-step, one-token-at-a-time processing of older networks) entirely, letting the whole input be processed at once instead of in sequence. Faster to train, parallelizable, and eventually more capable. The most consequential ML paper of the decade.

**2018** — BERT — Google's BERT (Bidirectional Encoder Representations from Transformers) achieves state-of-the-art on eleven NLP tasks simultaneously. It's trained by hiding random words in real text and having the model guess them from the words on both sides ("bidirectional," and "unsupervised" because no human had to label anything) on Wikipedia + BooksCorpus, followed by fine-tuning it further for each specific task. Transfer learning for NLP becomes the dominant paradigm.

**2018** — GPT-1 — OpenAI releases GPT (Generative Pre-trained Transformer). Unlike BERT, which reads a whole passage at once to fill in blanks, GPT reads only what came before and predicts the next token, one word at a time, left to right — a "causal" decoder because each prediction can only depend on what came earlier, never later. The architecture that eventually becomes GPT-4 and its successors.

**2019** — GPT-2 — OpenAI trains a 1.5B parameter autoregressive language model. The generated text is coherent enough that OpenAI initially withholds the full model, citing misuse risk. This decision is widely debated; the model is eventually released.

---

## 2020s: The LLM era

**2020** — GPT-3 — OpenAI releases GPT-3 (175B parameters). Few-shot prompting — providing a few examples in the context window rather than fine-tuning — proves surprisingly effective. The API business model for LLMs begins.

**2021** — GitHub Copilot (preview) — OpenAI Codex, fine-tuned on code, powers GitHub Copilot. The first mainstream AI coding assistant. By 2023, Copilot has millions of active users.

**2021** — DALL-E — OpenAI's DALL-E generates images from text descriptions using a transformer over image patches and text tokens. Multimodal generation becomes accessible.

**2022** — Stable Diffusion — Stability AI releases Stable Diffusion: a high-quality text-to-image diffusion model with open weights. Launches an ecosystem of image generation tools. Latent diffusion (operating in a compressed latent space rather than pixel space) makes generation viable on consumer hardware.

**2022** — InstructGPT / RLHF — OpenAI publishes InstructGPT: a method for aligning language models with human preferences using Reinforcement Learning from Human Feedback (RLHF). Models become dramatically more useful and less likely to produce harmful outputs. The technique becomes standard for all major assistant models.

**2022** — ChatGPT — OpenAI releases ChatGPT based on GPT-3.5 with RLHF fine-tuning. Reaches 100 million users in two months — the fastest consumer product adoption ever recorded at that point. Mainstreams large language models for non-technical users.

**2023** — GPT-4 — OpenAI's GPT-4 is released. Multimodal (text + images), significantly stronger reasoning, passes the Bar exam (early reports claimed top 10%; subsequent analysis places performance closer to median law school graduate). First model where capability differences from its predecessor are widely noted by professional users.

**2023** — Llama — Meta releases Llama and Llama 2 with open weights. Enables a wave of fine-tuned variants (Alpaca, Vicuna, Mistral) and on-premise deployment. Shifts the industry conversation toward open vs. closed model strategies.

**2023** — Claude — Anthropic releases Claude, trained with Constitutional AI (CAI): a self-critique and revision process that reduces the need for human labeling in alignment. Demonstrates that safety-oriented training can produce highly capable models.

**2023** — GPT-4V, Gemini — Multimodal becomes standard. GPT-4V adds vision; Google's Gemini is natively multimodal across text, images, audio, and video from the start.

**2024** — Reasoning models — OpenAI's o1, DeepSeek-R1, and subsequently Claude's extended thinking mode demonstrate that training models to produce explicit reasoning chains (chain-of-thought) before answering dramatically improves performance on mathematical and logical tasks.

**2024** — Long context — Context windows extend from 4K (GPT-3) to 8K (GPT-4) to 128K (GPT-4 Turbo), 200K (Claude 3), and 1M+ tokens (Gemini 1.5 Pro). Entire codebases or books fit in a single context.

**2025** — Agents enter production — Tool use, multi-step planning, and computer use capabilities mature. Agent frameworks (LangGraph, CrewAI, Claude's computer use) move from demos to production deployments at scale.
