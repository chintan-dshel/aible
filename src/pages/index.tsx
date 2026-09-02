import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <main
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '60vh',
          padding: '2rem',
          textAlign: 'center',
        }}>
        <h1 style={{fontSize: '3rem', marginBottom: '0.25rem'}}>Agentic Systems</h1>
        <p
          style={{
            fontSize: '1.4rem',
            fontWeight: 600,
            marginBottom: '1.25rem',
            opacity: 0.9,
          }}>
          Patterns and Teardowns
        </p>
        <p
          style={{
            fontSize: '1.1rem',
            maxWidth: '620px',
            marginBottom: '0.75rem',
            opacity: 0.85,
          }}>
          A short book on how to design a multi-agent AI system: eight design
          decisions you will hit in any of them, and teardowns of systems I
          built and run, read from their own source code.
        </p>
        <p
          style={{
            fontSize: '1rem',
            maxWidth: '580px',
            marginBottom: '2rem',
            opacity: 0.65,
          }}>
          For engineers and technical PMs who have never designed a multi-agent
          system and are about to. Underneath it sits a 45-page AI reference
          library, foundations to frontier.
        </p>
        <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center'}}>
          <Link className="button button--primary button--lg" to="/docs/intro">
            Start the book →
          </Link>
          <Link className="button button--secondary button--lg" to="/docs/foundations/">
            Browse the reference library
          </Link>
        </div>
      </main>
    </Layout>
  );
}
