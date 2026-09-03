---
sidebar_position: 4
title: World Models
description: Internal simulation, Dreamer-style models, and whether LLMs are world models — the ongoing debate.
---

# World Models

A world model is a system's internal representation of how reality works — one that lets it simulate what would happen if it took a given action, without actually taking it. Whether LLMs have anything like this is one of the most actively debated questions in AI.

## The problem it solves

Planning requires counterfactual reasoning: "if I do X, what happens?" A system that can only react to what it observes, but can't predict consequences, can't plan. World models close this gap by enabling a system to mentally simulate actions before committing to them — the basis of everything from chess engines to autonomous vehicle planning to human imagination.

For practitioners, the question matters because: systems with genuine world models should generalize better to novel situations, fail more predictably, and be easier to verify than systems that pattern-match without understanding.

## How it works under the hood

**Explicit world models (Dreamer-style).** In model-based reinforcement learning (RL, an approach where a system learns by taking actions and getting reward or penalty signals back), a world model is a learned function that predicts next state given current state and action. In plain terms, that's a next-state predictor: feed it "here's the situation, here's what I'm about to do," and it outputs "here's roughly what would happen." Dreamer (Hafner et al.) learns this predictor not in raw pixels but in a compact latent space — a compressed, numeric summary of a situation that the model itself learned, small enough to run thousands of hypothetical "what happens next" predictions quickly. This compressed representation is called a Recurrent State Space Model (RSSM): given an encoded observation, the model predicts the next latent state. The agent then plans by rolling out trajectories *in latent space* — thousands of simulated futures — and selecting the action sequence that leads to the best predicted outcome. This is never "imagining" in a conscious sense; it's iterated matrix multiplication (repeated numeric grid computations) over that learned compressed representation.

**The LLM world model debate.** Large language models, trained by repeatedly predicting the next word in real text, implicitly see vast descriptions of how the world works: physics, causality, social dynamics, geography. Do they learn a world model in the process?

Evidence for: GPT-4 and Claude can reason about counterfactuals, predict consequences of hypothetical actions, and answer questions about physical processes they've never been explicitly trained to answer. Probing experiments — training a small, separate classifier to read a model's internal numbers and see what information is recoverable from them — (Gurnee & Tegmark, 2023) found that those internal numbers directly and simply encode spatial and temporal coordinates ("linearly encode": recoverable by a simple weighted sum, not buried in some more convoluted pattern) — a property consistent with a learned world model.

Evidence against: LLMs fail systematically on simple tasks requiring reliable spatial reasoning (left/right, inside/outside), exact counting, and novel physical simulations (LeCun, 2022; Chollet, 2019). They are sensitive to prompt framing in ways a true world model wouldn't be. They often confabulate plausible-sounding but wrong consequences when asked about edge cases.

A prevalent view as of mid-2025 (contested — see LeCun in Going Deeper for the opposing argument): LLMs have *partial* world models — good enough to be useful for a broad range of reasoning tasks, but not the kind of robust, general, simulator-grade world models that would enable reliable autonomous agents without human oversight.

## Concrete example

This example shows the Dreamer-style architecture conceptually — not a full training loop (which requires an environment and thousands of GPU hours), but the prediction/planning structure. The first function asks an LLM to act as a crude world model, predicting what happens after an action; the second uses that prediction repeatedly to score candidate actions against a goal and pick the best one — a language-domain stand-in for the latent-space rollout Dreamer does numerically:

```python
import anthropic
import json

client = anthropic.Anthropic()


def llm_world_model_probe(scenario: str, action: str) -> dict:
    """
    Use an LLM as a world model: given a scenario and action,
    predict what happens. Returns predicted next state and confidence.

    This illustrates the concept of a world model as a predict-next-state function.
    LLM-as-world-model is useful for language-domain reasoning but unreliable
    for precise physical simulation.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=(
            "You are a world model. Given a scenario and an action, predict the most likely "
            "next state. Be specific and concrete. Rate your confidence from 0.0 to 1.0 — "
            "use low confidence for physics-heavy, spatial, or numerically precise predictions."
        ),
        messages=[{
            "role": "user",
            "content": json.dumps({
                "current_state": scenario,
                "action": action,
                "predict": "next_state",
            }),
        }],
    )
    return {
        "scenario": scenario,
        "action": action,
        "predicted_state": response.content[0].text,
    }


def latent_space_planner(
    goal: str,
    initial_state: str,
    candidate_actions: list[str],
    horizon: int = 3,
) -> list[str]:
    """
    Simple lookahead planner using LLM world model predictions.
    Simulates action sequences up to `horizon` steps and selects
    the sequence predicted to best achieve the goal.

    This is the planning-in-imagination loop that Dreamer does in latent space;
    here we do it in language space, which is looser but illustrates the concept.
    """
    best_score = -1.0
    best_sequence: list[str] = []

    # Enumerate single-step candidates (in practice: beam search or MCTS)
    for action in candidate_actions:
        prediction = llm_world_model_probe(initial_state, action)

        # Score the predicted next state against the goal
        score_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": (
                    f"Goal: {goal}\n"
                    f"Predicted state after action '{action}':\n{prediction['predicted_state']}\n\n"
                    f"How close is this state to the goal? Reply with only a decimal from 0.0 to 1.0."
                ),
            }],
        )
        try:
            score = float(score_response.content[0].text.strip())
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.0

        if score > best_score:
            best_score = score
            best_sequence = [action]

    return best_sequence


# Probe: test whether the LLM encodes causal relationships
scenarios = [
    ("A glass of water is on a table. Someone bumps the table.", "The glass falls off the edge."),
    ("A city raises its minimum wage.", "Employment levels change."),
    ("A user deletes a git branch with unmerged commits.", "The commits become unreachable."),
]

for scenario, action in scenarios:
    result = llm_world_model_probe(scenario, action)
    print(f"Scenario: {scenario}")
    print(f"Action: {action}")
    print(f"Prediction: {result['predicted_state'][:200]}\n")
```

**What the Dreamer RSSM actually looks like (conceptual pseudocode):**

```python
# Not executable — illustrates the Dreamer architecture
class RSSM:
    """Recurrent State Space Model — the learned world model in Dreamer."""

    def observe(self, prev_state, action, observation):
        """Encode observation + prior state into new posterior state."""
        h = self.gru(prev_state["h"], action)          # deterministic path
        z = self.encoder(observation, h)                # stochastic posterior
        return {"h": h, "z": z}

    def imagine(self, state, action):
        """Predict next state WITHOUT an observation (pure imagination)."""
        h = self.gru(state["h"], action)
        z = self.prior(h)                              # prior, not posterior
        return {"h": h, "z": z}

    def decode(self, state):
        """Reconstruct observation from latent state (for training)."""
        return self.decoder(state["h"], state["z"])

    def plan(self, initial_state, horizon=15):
        """Simulate trajectories in imagination and return best action sequence."""
        # Actor network proposes actions; value network scores states
        # All computation happens in latent space — no environment calls
        ...
```

## When to use it / when not to

**LLM as world model (language domain):**
- Valid for reasoning tasks where the "world" is text, code, logic, or social dynamics
- Useful in planning loops where you want to evaluate candidate actions before committing (agentic systems, decision support)
- Unreliable for physics-precise simulation, exact spatial reasoning, or novel edge cases outside the training distribution — use [[Confidence Estimation]] techniques to detect when the model is operating outside its reliable range

**Dreamer-style explicit world models:**
- Production-ready only for constrained, well-defined environments (game playing, robotic manipulation in controlled settings)
- Not yet practical for open-ended real-world deployment — require vast environment interaction to train, and fail on distribution shift (real-world inputs that look different from anything seen during training)

**The honest answer:** If your task needs reliable counterfactual prediction over a structured physical domain, use a physics simulator or domain model, not an LLM. If your task involves reasoning about likely consequences in the language domain (business decisions, social dynamics, code behavior), LLMs as world models are useful with appropriate uncertainty.

## Main tools and libraries

| Tool | Role |
|------|------|
| [Dreamer V3](https://github.com/danijar/dreamerv3) | Reference implementation — model-based RL with learned world models |
| [Gymnasium](https://gymnasium.farama.org/) | Standard RL environment interface for training world models |
| `torch` / `jax` | World model training requires custom neural architecture code |
| [BabyAI](https://github.com/mila-iqia/babyai) | Structured language + grid environment for studying language-grounded world models |
| Anthropic Claude API | LLM-as-world-model for language-domain planning |

## Common failure modes and gotchas

**Compounding simulation error.** Every step of a learned world model introduces prediction error. Over long horizons, these errors compound — the agent plans for an imagined world that diverges from reality. Dreamer addresses this with imagination rollout length limits and regular re-grounding to real observations.

**Distribution shift breaks the model.** A world model trained in environment A fails in environment B, even if B seems similar. LLMs have this problem too — they fail on physical edge cases outside their training distribution. Never assume a learned world model generalizes cleanly.

**Overconfidence in LLM world models.** LLMs produce fluent, confident-sounding predictions for physical scenarios they can't actually reason about correctly. Treat LLM world model predictions as soft priors — tentative starting estimates to be updated with real evidence, not settled facts — not ground truth (the actual, verified correct answer). Ask for uncertainty estimates.

**The "Chinese Room" failure mode** — named for a thought experiment about whether following rules that produce the right output means you actually understand what you're doing. An LLM that correctly predicts "the glass breaks when it falls" may have learned this as a statistical pattern from text rather than understanding glass fragility. The prediction may be right for the common case but wrong for physically unusual scenarios (e.g., a glass falling on carpet vs. tile). Test edge cases deliberately.

## Project ideas

- **World model probe test suite**: design 50 scenarios across domains (physics, social dynamics, code behavior) where you know the correct outcome. Test Claude and other models; compare accuracy across domains and use it to calibrate when to trust LLM world model predictions.
- **Latent space planner**: implement the lookahead planning loop above for a specific domain (e.g., business strategy: "if we lower our price, what happens to demand?"). Compare plans from the LLM world model against historical data.
- **Train a Dreamer agent**: use the Dreamer V3 implementation on a Gymnasium environment. Visualize the latent space trajectories to understand what the world model has learned.

## Going deeper

- [Hafner et al., "Mastering Diverse Domains with World Models" (DreamerV3)](https://arxiv.org/abs/2301.04104) — the reference model-based RL architecture
- [Ha & Schmidhuber, "World Models" (2018)](https://arxiv.org/abs/1803.10122) — foundational paper; accessible and visual
- [Gurnee & Tegmark, "Language Models Represent Space and Time" (2023)](https://arxiv.org/abs/2310.02207) — probing LLMs for world model representations
- [LeCun, "A Path Towards Autonomous Machine Intelligence"](https://openreview.net/pdf?id=BZ5a1r-kVsf) — argues current LLMs can't be world models; proposes a different architecture (JEPA)
- [Chollet, "On the Measure of Intelligence" (2019)](https://arxiv.org/abs/1911.01547) — introduces the ARC benchmark; argues LLMs achieve pattern-matching rather than general intelligence, with direct implications for the world model debate
- [[Long-Horizon Agents]] — planning systems that need reliable world model predictions to work
