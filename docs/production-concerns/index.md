---
sidebar_position: 1
title: 'Reference: Production'
sidebar_label: 'Reference: Production'
description: What changes when AI moves from demo to production — reliability, monitoring, deployment patterns, and graceful failure.
---

# Reference: Production

This section covers what's different about deploying AI systems compared to conventional software: non-determinism, confidence without calibration, silent failures, and the unique challenges of monitoring something that outputs free text.

## Reading order — taking a system to production

1. [Reliability](./reliability) — circuit breakers, retries, backoff
2. [Graceful Degradation](./graceful-degradation) — fallback hierarchy
3. [Monitoring](./monitoring) — LLM-as-judge, structural metrics, alerts
4. [Deployment Patterns](./deployment-patterns) — shadow mode, canary, A/B
5. [Confidence Estimation](./confidence-estimation) — when to trust the model
6. [Fallbacks](./fallbacks) — multi-model fallback chains

Supplement with [Observability](../meta-infrastructure/observability), [Evals](../meta-infrastructure/evals), and [Output Validation](../meta-infrastructure/output-validation) in Reference: Infrastructure.

## Topics

- [Reliability](./reliability) — retry logic, idempotency, circuit breakers for LLM calls
- [Graceful Degradation](./graceful-degradation) — fallback content, confidence thresholds, human escalation
- [Confidence Estimation](./confidence-estimation) — logprobs, self-consistency, calibration
- [Fallbacks](./fallbacks) — model fallback chains and deterministic fallback content
- [Monitoring](./monitoring) — drift detection, quality regression, alerting
- [Deployment Patterns](./deployment-patterns) — shadow mode, A/B, canary, blue-green for AI systems
