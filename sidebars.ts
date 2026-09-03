import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Explicit, fully-nested sidebar. Reading order, front to back:
 *
 *   Start Here
 *   AI Foundations              -- mechanism: what a model is doing
 *   Agentic Building Blocks     -- the pieces every agentic system assembles from
 *   Agentic Architecture Patterns -- the book, part 1: 8 design-decision chapters
 *   Teardowns: Systems I Built  -- the book, part 2: 4 real systems, read from source
 *   Infrastructure and Tooling  -- entered from a chapter, not read cold
 *   Running in Production
 *   The Frontier
 *   Glossary
 *
 * No "Part 1"/"Part 2" wording anywhere -- both book sections are real
 * nested categories now, not flat links next to their own chapters.
 * Teardowns are listed by system name, not by a global chapter number
 * (there is no "9. ProjectOS" -- the book's 1-8 numbering belongs to the
 * pattern chapters only).
 *
 * Two pages are cross-listed into Agentic Building Blocks because they
 * are prerequisite tooling/UX for the patterns chapters, not deep
 * reference or speculative material: Orchestration Frameworks (lives at
 * meta-infrastructure/orchestration-frameworks) and Long-Horizon Agents
 * (lives at aspirational/long-horizon-agents). Their file location and
 * URL are unchanged -- this file only controls where they appear in the
 * nav. Each page's own section index no longer re-lists it, so it
 * appears exactly once in the whole sidebar.
 */
const sidebars: SidebarsConfig = {
  mainSidebar: [
    {type: 'doc', id: 'intro', label: 'Start Here'},

    {
      type: 'category',
      label: 'AI Foundations',
      link: {type: 'doc', id: 'foundations/index'},
      items: [
        'foundations/what-is-ai',
        'foundations/how-llms-work',
        'foundations/neural-networks',
        'foundations/transformers',
        'foundations/attention',
        'foundations/embeddings',
        'foundations/tokenization',
        'foundations/training-vs-inference',
        'foundations/history-timeline',
        'foundations/history-narrative',
      ],
    },

    {
      type: 'category',
      label: 'Agentic Building Blocks',
      link: {type: 'doc', id: 'core-building-blocks/index'},
      items: [
        'core-building-blocks/prompting',
        'core-building-blocks/structured-outputs',
        'core-building-blocks/function-calling',
        'core-building-blocks/rag',
        'core-building-blocks/multi-agent-systems',
        'meta-infrastructure/orchestration-frameworks',
        'core-building-blocks/memory-architectures',
        'aspirational/long-horizon-agents',
      ],
    },

    {
      type: 'category',
      label: 'Agentic Architecture Patterns',
      link: {type: 'doc', id: 'patterns/index'},
      items: [
        'patterns/orchestration-vs-autonomy',
        'patterns/stage-gates',
        'patterns/state-machines',
        'patterns/llm-as-judge',
        'patterns/qa-pipelines',
        'patterns/cost-aware-model-routing',
        'patterns/memory-layers',
        'patterns/failure-modes-and-attack-surfaces',
      ],
    },

    {
      type: 'category',
      label: 'Teardowns: Systems I Built',
      link: {type: 'doc', id: 'teardowns/index'},
      items: [
        'teardowns/projectos',
        'teardowns/lyceum',
        'teardowns/second-brain',
      ],
    },

    {
      type: 'category',
      label: 'Infrastructure and Tooling',
      link: {type: 'doc', id: 'meta-infrastructure/index'},
      items: [
        'meta-infrastructure/evals',
        'meta-infrastructure/observability',
        'meta-infrastructure/guardrails',
        'meta-infrastructure/prompt-injection',
        'meta-infrastructure/output-validation',
        'meta-infrastructure/vector-databases',
        'meta-infrastructure/caching',
        'meta-infrastructure/mcp',
        'meta-infrastructure/cost-tracking',
        'meta-infrastructure/latency-optimization',
        'meta-infrastructure/model-routing',
        'meta-infrastructure/model-distillation',
        'meta-infrastructure/synthetic-data',
        'meta-infrastructure/document-processing',
        'meta-infrastructure/knowledge-graphs',
        'meta-infrastructure/red-teaming',
        'meta-infrastructure/audit-logs',
        'meta-infrastructure/access-control',
      ],
    },

    {
      type: 'category',
      label: 'Running in Production',
      link: {type: 'doc', id: 'production-concerns/index'},
      items: [
        'production-concerns/reliability',
        'production-concerns/graceful-degradation',
        'production-concerns/confidence-estimation',
        'production-concerns/fallbacks',
        'production-concerns/monitoring',
        'production-concerns/deployment-patterns',
      ],
    },

    {
      type: 'category',
      label: 'The Frontier',
      link: {type: 'doc', id: 'aspirational/index'},
      items: [
        'aspirational/agentic-computer-use',
        'aspirational/world-models',
        'aspirational/multimodal-frontier',
        'aspirational/on-device-ai',
        'aspirational/novel-interaction',
        'aspirational/ai-safety',
      ],
    },

    {type: 'doc', id: 'glossary', label: 'Glossary'},
  ],
};

export default sidebars;
