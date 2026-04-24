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
        <h1 style={{fontSize: '3rem', marginBottom: '0.5rem'}}>Aible</h1>
        <p
          style={{
            fontSize: '1.2rem',
            maxWidth: '580px',
            marginBottom: '2rem',
            opacity: 0.8,
          }}>
          {siteConfig.tagline}
        </p>
        <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center'}}>
          <Link className="button button--primary button--lg" to="/docs/intro">
            Start reading →
          </Link>
          <Link className="button button--secondary button--lg" to="/docs/foundations/">
            Jump to Foundations
          </Link>
        </div>
      </main>
    </Layout>
  );
}
