import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Agentic Systems',
  tagline: 'How to design a multi-agent AI system — patterns, teardowns of systems I built, and the AI reference underneath.',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://chintan-dshel.github.io',
  baseUrl: '/aible/',

  organizationName: 'chintan-dshel',
  projectName: 'aible',

  onBrokenLinks: 'warn',

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  themes: ['@docusaurus/theme-mermaid'],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      type: 'text/css',
      crossorigin: 'anonymous',
    },
  ],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['en'],
        docsDir: 'docs',
        indexPages: true,
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Agentic Systems',
      items: [
        {
          type: 'dropdown',
          label: 'The Book',
          position: 'left',
          items: [
            {to: '/docs/patterns/', label: 'Agentic Architecture Patterns'},
            {to: '/docs/teardowns/', label: 'Teardowns: Systems I Built'},
          ],
        },
        {
          type: 'dropdown',
          label: 'Reference',
          position: 'left',
          items: [
            {to: '/docs/foundations/', label: 'AI Foundations'},
            {to: '/docs/core-building-blocks/', label: 'Agentic Building Blocks'},
            {to: '/docs/meta-infrastructure/', label: 'Infrastructure and Tooling'},
            {to: '/docs/production-concerns/', label: 'Running in Production'},
            {to: '/docs/aspirational/', label: 'The Frontier'},
          ],
        },
        {
          to: '/docs/glossary',
          label: 'Glossary',
          position: 'left',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright: `Agentic Systems: Patterns and Teardowns. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'typescript', 'rust'],
    },
    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
      options: {
        fontFamily:
          'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        fontSize: 16,
        themeVariables: {
          fontSize: '16px',
          fontFamily:
            'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        },
        flowchart: {
          useMaxWidth: false,
          htmlLabels: true,
          curve: 'basis',
          nodeSpacing: 55,
          rankSpacing: 75,
          padding: 14,
          diagramPadding: 12,
          wrappingWidth: 220,
        },
        sequence: {
          useMaxWidth: false,
          wrap: true,
          width: 170,
          height: 48,
          actorMargin: 55,
          boxMargin: 12,
          messageMargin: 38,
          mirrorActors: false,
          diagramMarginX: 24,
          diagramMarginY: 14,
          actorFontSize: 15,
          messageFontSize: 13,
          noteFontSize: 13,
          wrapPadding: 12,
        },
        state: {
          useMaxWidth: false,
          nodeSpacing: 55,
          rankSpacing: 75,
          padding: 14,
          titleTopMargin: 12,
        },
      },
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
