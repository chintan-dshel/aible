---
sidebar_position: 4
title: History — Narrative
description: The story of how we got here — key inflection points told with actual substance.
---

# History — Narrative

The history of AI is not a steady march toward capability. It's a sequence of confident predictions, embarrassing failures, quiet rebuildings, and then eruptions that surprise even the people doing the work. Understanding the pattern helps you read the current moment more clearly.

For the dense chronological record, see [History — Timeline](history-timeline.md).

---

## The founding bet (1950–1969)

The field begins with a philosophical question dressed as an engineering problem. Alan Turing's 1950 paper "Computing Machinery and Intelligence" asks whether machines can think and immediately sidesteps the question: he proposes the Imitation Game instead, reframing "thinking" as behavioral indistinguishability from a human in conversation. It's a clever move — it shifts the debate from metaphysics to measurement — but it embeds a confusion that the field still hasn't fully resolved. Behaving like something intelligent and being intelligent are not obviously the same.

The 1956 Dartmouth Conference that coined the term "Artificial Intelligence" was organized around a specific bet: that "every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it." The conference participants expected to make significant progress on the problem in one summer. They did not. But they named the field and set its ambitions, and those ambitions proved sticky.

The decade that followed was genuinely productive in narrow ways. LISP was invented. Expert systems like DENDRAL showed that encoding domain knowledge explicitly could produce expert-level performance on well-scoped tasks. ELIZA demonstrated that humans would project understanding onto any system that mirrored their words back — an observation about human psychology rather than machine capability, though this distinction was often lost.

The first hard wall arrived in 1969 when Minsky and Papert published *Perceptrons*, proving mathematically that single-layer neural networks cannot learn XOR — a pattern that no single straight dividing line can separate, however you draw it. The proof was correct. The interpretation — that neural networks were therefore a dead end — was not, but it was influential enough to divert funding away from neural approaches for years.

---

## Expert systems and the first winter (1970–1987)

The dominant paradigm through the 1970s and most of the 1980s was symbolic AI: the idea that intelligence is manipulation of symbols according to rules, and that the rules can be written by hand. Expert systems — programs encoding the knowledge of domain specialists — became a substantial industry. XCON, deployed by Digital Equipment Corporation in 1980, saved the company an estimated $40 million per year configuring VAX computer orders. Dozens of similar systems followed in medicine, finance, and engineering.

The problem became apparent around 1987: expert systems are brittle. They work precisely within their knowledge base and fail completely outside it. Maintaining them as domains evolved was expensive and slow. Rules written for one context couldn't transfer to adjacent contexts. The systems that had been confidently described as "nearly thinking" turned out to require constant manual intervention to stay accurate.

The AI industry, which had attracted substantial venture capital through the early 1980s, lost most of it by the end of the decade. What makes this period instructive: the failure wasn't from wrong ideas about what intelligence requires. Expert systems failed because encoding knowledge by hand doesn't scale. You can write rules for a narrow domain. You cannot write rules for everything a competent person knows. The field needed a different approach to acquiring knowledge — one that didn't require experts to articulate their expertise explicitly.

---

## The quiet rebuilding (1986–2011)

While the expert systems industry was collapsing, a smaller community was doing something different. In 1986, Rumelhart, Hinton, and Williams published the backpropagation paper in *Nature* — not inventing the algorithm but framing it clearly enough that neural network researchers could use it. Backpropagation is the algorithm that lets a network learn from being wrong: run an example through, compare the output to the right answer, then work backward through the network adjusting each internal number (a "parameter") by how much it contributed to the error. The key insight underneath it: if a function's error can be measured in a way smooth enough to calculate a direction of improvement from ("differentiable"), you can nudge it step by step toward a better answer — that stepwise nudging is what "gradient descent" means — and that's true whether the function is small or, eventually, has billions of adjustable parameters.

Yann LeCun applied this to handwritten digit recognition with convolutional networks in the late 1980s. The results were good enough to deploy — AT&T used LeCun's network to read zip codes on mail. But it required weeks of training on hardware that cost hundreds of thousands of dollars. The approach was correct; the infrastructure wasn't ready.

Throughout the 1990s and 2000s, kernel methods — techniques for drawing a boundary between categories that can bend to fit non-straight patterns, rather than only ever being a straight line — dominated machine learning benchmarks. SVMs (support vector machines, the best-known method built on this idea) had elegant theoretical foundations, worked well in practice, and didn't require the finicky training that neural networks did. A smaller group of neural network researchers worked on problems that the mainstream regarded as marginal — sequence modeling, speech, language. They were quietly accumulating the ideas that would matter later.

Two subplots ran through this period. First, GPUs: originally designed for rendering video game graphics, they turned out to be extraordinarily well-suited for the parallel matrix multiplication that neural networks require. NVIDIA's CUDA (2007) — a set of tools that let researchers write general-purpose code for a GPU, not just graphics code — made GPU programming accessible to researchers. The same networks that would have taken months to train in 2005 took days in 2010. Second, data: the internet was accumulating massive text corpora; Fei-Fei Li and her team spent three years assembling ImageNet — 14 million labeled images. The training data needed to make large neural networks work was becoming available.

---

## The eruption (2012–2017)

AlexNet in 2012 is the clean before/after. Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton entered the ImageNet challenge with a deep convolutional network trained on two consumer gaming GPUs. They achieved a top-5 error rate of 15.3%. Second place was 26.2%. The gap was so large that the field didn't debate whether this was noise.

Within two years, essentially every major company with a computer vision problem had switched to deep learning. The pattern accelerated: word2vec (2013) showed that word meanings could be encoded as geometric relationships in high-dimensional space — represent each word as a list of numbers, and the arithmetic on those lists lines up with meaning: "king" minus "man" plus "woman" lands close to "queen." GANs (2014) showed that networks could generate realistic images by pitting a generator against a discriminator — one network learns to fake images, a second learns to catch the fakes, and the competition between them makes the fakes better over time. Residual connections (2015) removed the practical depth limit on networks — before this, stacking too many layers made a network worse, not better, because the training signal degraded on its way back through so many layers; residual connections give that signal a shortcut path around each layer — enabling 150-layer architectures.

AlphaGo in 2016 marked the end of the era when Go could be used as an argument that there are things machines fundamentally cannot do. The argument had been that Go's branching factor made brute-force search intractable, and that human intuition about "good positions" couldn't be encoded. AlphaGo learned that intuition from 30 million human games and then refined it through self-play. When it beat Lee Sedol 4–1, Go professionals described some of its moves as beautiful and occasionally alien.

The transformer paper in 2017 — "Attention Is All You Need" — replaced recurrent connections with attention mechanisms and changed what was computationally tractable for language. Recurrent networks had to process text sequentially, one token (roughly, one word or word-fragment) at a time; attention processes the entire sequence simultaneously. It also removed the information bottleneck of squeezing long sequences through a single hidden state vector — the older approach's one fixed-size running summary, which everything the network had read so far had to be compressed into before it could use any of it. The implications took another year to become fully apparent.

---

## Language becomes the medium (2018–2022)

BERT in 2018 demonstrated something that changed how the field thought about capability. You could train a model to predict masked words in Wikipedia text — a self-supervised task requiring no human annotation — and then fine-tune that model to achieve state-of-the-art performance on tasks ranging from question answering to sentiment analysis to named entity recognition. Transfer learning for language worked the way it had worked for images after AlexNet.

GPT-3 in 2020 introduced a phenomenon that few had predicted: few-shot learning. Give the model three examples of the task in its context window, and it performs the task with no additional training. The model had learned, from predicting the next token on a large enough corpus, something like a general-purpose task adapter. It could write code, translate languages, answer questions about chemistry — not perfectly, but well enough to be useful, without being explicitly trained to do any of these things.

This is the point where the field's self-understanding had to revise. The implicit assumption behind most AI research had been that different capabilities required different architectures and different training regimes. GPT-3 suggested that a sufficiently large model trained on sufficiently much text would develop a wide range of capabilities as emergent side effects.

ChatGPT in 2022 added a layer that GPT-3 lacked: Reinforcement Learning from Human Feedback, making the model cooperative and helpful rather than merely completing text. The difference between a text completion engine and an assistant turns out to matter enormously for usability. ChatGPT reached 100 million users in two months.

---

## The current era (2023–present)

The distinguishing features of the current moment are scale, multimodality, and the beginning of agentic capability.

**Scale** now produces qualitative differences. GPT-4 passes the bar exam — early reports claimed top 10% of human test-takers, though subsequent analysis suggests performance is closer to the median of actual law school graduates; the gap between test-taker and law-school-graduate populations makes the comparison contested. It solves competition mathematics problems that required doctoral-level reasoning to produce a few years ago. Whether this constitutes "real" reasoning or "sophisticated pattern matching" is a question philosophers of mind argue about; practitioners mostly observe that the outputs are useful and getting more so.

**Multimodality** is no longer a research demo. GPT-4V, Gemini, and Claude process images, audio, and text within the same model. Stable Diffusion and its successors generate images from text descriptions on consumer hardware. The boundary between modalities is dissolving.

**Agents** — models that can use tools, take multi-step actions, and interact with the world beyond generating text — are moving from research to production. Models browse the web, write and execute code, interact with APIs, and operate desktop interfaces. The reliability is imperfect; the trajectory is not.

What's genuinely unknown is where the current scaling approach runs into its limits. The standard critique — that large language models don't "really understand" and therefore can't generalize beyond their training distribution — has been made confidently before each capability jump and has so far underestimated what training on enough diverse data achieves. It may be correct eventually. The honest position is that nobody knows where the ceiling is.

The pattern that repeats in both directions: confident claims about limits, followed by the limits being exceeded, followed by new claims about the next set of limits — but also confident claims about capabilities that turn out to be brittle, inconsistent, or illusory under careful evaluation. Staying calibrated means holding beliefs lightly about what these systems can and cannot do — updating when you see evidence, rather than fitting evidence to a prior conclusion in either direction.
