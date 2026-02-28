import { useState } from "react";
import { parseCv, generateApplication, pollResults } from "../api/applications";
import type { ResultsResponse } from "../api/applications";
import { CVUploader } from "../components/CVUploader";
import { JobDescInput } from "../components/JobDescInput";
import { ProgressIndicator } from "../components/ProgressIndicator";
import ResultsViewer from "../components/ResultsViewer";
import DiffViewer from "../components/DiffViewer";
import DownloadButton from "../components/DownloadButton";

type Step = "upload" | "describe" | "processing" | "results";

export function Home() {
  const [step, setStep] = useState<Step>("upload");
  const [cvFile, setCvFile] = useState<File | null>(null);
  const [jobDesc, setJobDesc] = useState("");
  const [stage, setStage] = useState<"idle" | "parsing_cv" | "generating_letter" | "auditing" | "done" | "failed">("idle");
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [appId, setAppId] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const apiBase = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

  async function handleGenerate() {
    if (!cvFile || !jobDesc.trim()) return;
    setStep("processing");  // ← add this back
    setError(null);

    try {
      setStage("parsing_cv");
      const parsed = await parseCv(cvFile);

      setStage("generating_letter");
      const gen = await generateApplication({ cv_text: parsed.parsed_text, job_description: jobDesc });
      setAppId(gen.request_id);
      setAccessToken(gen.access_token);

      setStage("auditing");
      const final = await pollResults(gen.request_id, gen.access_token, { intervalMs: 1500, timeoutMs: 120_000 });
      setResults(final);
      setStage(final.status === "done" ? "done" : "failed");
      setStep("results");
    } catch (e: any) {
      setError(e?.message ?? "Something went wrong");
      setStage("failed");
    }
  }

  const audit = results?.audit_report?.verifications?.map((a, i) => ({
    id: String(i),
    claim: a.claim,
    supported: a.supported,
    confidence: a.confidence,
    source: a.source,
  })) ?? [];

  const suggestions = results?.cv_suggestions
    ?.filter((s) => s.before && s.after)
    .map((s, i) => ({
      id: String(i),
      section: `Suggestion ${i + 1}`,
      before: s.before,
      after: s.after,
    })) ?? [];

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="app-header">
        <div className="header-inner">
          <div className="logo-mark">
            <span className="logo-icon">◈</span>
            <span className="logo-text">CVForge</span>
          </div>
          <nav className="header-nav">
            <span className="nav-tag">AI-Powered</span>
          </nav>
        </div>
      </header>

      {/* Hero */}
      {step === "upload" && (
        <section className="hero-section">
          <div className="hero-bg-grid" />
          <div className="hero-content">
            <div className="hero-eyebrow">Intelligent Application Engine</div>
            <h1 className="hero-title">
              Your CV,<br />
              <em>elevated.</em>
            </h1>
            <p className="hero-sub">
              Upload your CV and a job description. We'll generate a tailored cover letter,
              suggest improvements, and audit every claim.
            </p>
            <div className="hero-uploader">
              <CVUploader maxSizeMb={5} onFileSelected={(f) => { setCvFile(f); setStep("describe"); }} />
            </div>
          </div>
          <div className="hero-decorations">
            <div className="deco-ring deco-ring-1" />
            <div className="deco-ring deco-ring-2" />
            <div className="deco-orb" />
          </div>
        </section>
      )}

      {/* Job Description Step */}
      {step === "describe" && (
        <section className="step-section">
          <div className="step-container">
            <div className="step-header">
              <button className="back-btn" onClick={() => setStep("upload")}>← Back</button>
              <div className="file-badge">
                <span className="file-icon">📄</span>
                <span>{cvFile?.name}</span>
              </div>
            </div>
            <h2 className="step-title">Paste the job description</h2>
            <p className="step-sub">We'll tailor your application to match the role perfectly.</p>
            <div className="jd-input-wrap">
              <JobDescInput value={jobDesc} onChange={setJobDesc} />
            </div>
            <button
              className="generate-btn"
              disabled={!jobDesc.trim()}
              onClick={handleGenerate}
            >
              <span>Generate Application</span>
              <span className="btn-arrow">→</span>
            </button>
          </div>
        </section>
      )}

      {/* Processing */}
      {step === "processing" && (
        <section className="processing-section">
          <div className="processing-container">
            <div className="processing-pulse">
              <div className="pulse-ring" />
              <div className="pulse-core">◈</div>
            </div>
            <h2 className="processing-title">Crafting your application…</h2>
            <p className="processing-sub">This takes about 30–60 seconds</p>
            <div className="progress-wrap">
              <ProgressIndicator stage={stage} />
            </div>
            {error && (
              <>
                <p className="error-msg">{error}</p>
                <div className="error-actions">
                  <button className="retry-btn" onClick={handleGenerate}>↺ Try again</button>
                  <button className="back-btn" onClick={() => { setStep("describe"); setStage("idle"); setError(null); }}>← Edit inputs</button>
                </div>
              </>
            )}
          </div>
        </section>
      )}

      {/* Results */}
      {(stage === "done" || stage === "failed") && results && (
        <section className="results-section">
          <div className="results-container">
            <div className="results-header">
              <div>
                <h2 className="results-title">Application Ready</h2>
                <p className="results-sub">Review, refine, and download your materials below.</p>
              </div>
              {appId && accessToken && (
                <DownloadButton applicationId={appId} accessToken={accessToken} apiBaseUrl={apiBase} />
              )}
            </div>

            <div className="results-grid">
              {results.cover_letter && (
                <div className="result-card result-card--wide">
                  <ResultsViewer
                    coverLetter={results.cover_letter}
                    audit={audit}
                  />
                </div>
              )}
              {suggestions.length > 0 && (
                <div className="result-card">
                  <DiffViewer suggestions={suggestions} />
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #0c0e14;
          --bg-2: #12151e;
          --bg-3: #1a1f2e;
          --border: rgba(255,255,255,0.08);
          --border-bright: rgba(255,255,255,0.15);
          --text: #e8eaf0;
          --text-muted: #7a8099;
          --accent: #c9a96e;
          --accent-dim: rgba(201,169,110,0.15);
          --accent-glow: rgba(201,169,110,0.3);
          --green: #4ade80;
          --red: #f87171;
          --radius: 12px;
          --radius-lg: 20px;
          --font-display: 'Playfair Display', Georgia, serif;
          --font-body: 'DM Sans', system-ui, sans-serif;
        }

        body { background: var(--bg); color: var(--text); font-family: var(--font-body); }

        .app-shell { min-height: 100vh; display: flex; flex-direction: column; }

        /* Header */
        .app-header {
          position: fixed; top: 0; left: 0; right: 0; z-index: 100;
          background: rgba(12,14,20,0.85);
          backdrop-filter: blur(20px);
          border-bottom: 1px solid var(--border);
        }
        .header-inner {
          max-width: 1100px; margin: 0 auto;
          padding: 0 24px; height: 64px;
          display: flex; align-items: center; justify-content: space-between;
        }
        .logo-mark { display: flex; align-items: center; gap: 10px; }
        .logo-icon { color: var(--accent); font-size: 20px; }
        .logo-text { font-family: var(--font-display); font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
        .nav-tag {
          font-size: 11px; font-weight: 500; letter-spacing: 1.5px;
          text-transform: uppercase; color: var(--accent);
          border: 1px solid var(--accent-dim); padding: 4px 10px; border-radius: 999px;
        }

        /* Hero */
        .hero-section {
          position: relative; overflow: hidden;
          min-height: 100vh; display: flex; align-items: center;
          padding: 100px 24px 60px;
        }
        .hero-bg-grid {
          position: absolute; inset: 0;
          background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
          background-size: 60px 60px;
          mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
        }
        .hero-content {
          position: relative; z-index: 2;
          max-width: 680px; margin: 0 auto; text-align: center;
        }
        .hero-eyebrow {
          display: inline-block;
          font-size: 11px; font-weight: 600; letter-spacing: 2px;
          text-transform: uppercase; color: var(--accent);
          background: var(--accent-dim); border: 1px solid rgba(201,169,110,0.2);
          padding: 6px 16px; border-radius: 999px; margin-bottom: 28px;
        }
        .hero-title {
          font-family: var(--font-display);
          font-size: clamp(52px, 8vw, 96px);
          line-height: 1.0; letter-spacing: -2px;
          color: var(--text); margin-bottom: 24px;
        }
        .hero-title em {
          font-style: italic; color: var(--accent);
        }
        .hero-sub {
          font-size: 17px; line-height: 1.7; color: var(--text-muted);
          max-width: 480px; margin: 0 auto 48px;
        }
        .hero-uploader { max-width: 520px; margin: 0 auto; }
        .hero-decorations { position: absolute; inset: 0; z-index: 1; pointer-events: none; }
        .deco-ring {
          position: absolute; border-radius: 50%;
          border: 1px solid rgba(201,169,110,0.1);
        }
        .deco-ring-1 {
          width: 600px; height: 600px;
          right: -200px; top: -200px;
          animation: spin 40s linear infinite;
        }
        .deco-ring-2 {
          width: 400px; height: 400px;
          right: -100px; top: -100px;
          border-color: rgba(201,169,110,0.06);
          animation: spin 25s linear infinite reverse;
        }
        .deco-orb {
          position: absolute;
          width: 300px; height: 300px;
          right: 0; top: 50%;
          transform: translateY(-50%);
          background: radial-gradient(circle, rgba(201,169,110,0.08) 0%, transparent 70%);
          border-radius: 50%;
        }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

        /* Steps */
        .step-section, .processing-section, .results-section {
          min-height: 100vh; padding: 100px 24px 60px;
          display: flex; align-items: flex-start; justify-content: center;
        }
        .step-container, .processing-container, .results-container {
          width: 100%; max-width: 720px;
        }
        .results-container { max-width: 1000px; }

        .step-header { display: flex; align-items: center; gap: 16px; margin-bottom: 36px; }
        .back-btn {
          background: none; border: 1px solid var(--border);
          color: var(--text-muted); padding: 8px 16px;
          border-radius: var(--radius); font-family: var(--font-body);
          font-size: 14px; cursor: pointer; transition: all 0.2s;
        }
        .back-btn:hover { border-color: var(--border-bright); color: var(--text); }
        .file-badge {
          display: flex; align-items: center; gap: 8px;
          background: var(--bg-3); border: 1px solid var(--border);
          padding: 8px 14px; border-radius: var(--radius);
          font-size: 13px; color: var(--text-muted);
        }

        .step-title {
          font-family: var(--font-display); font-size: 40px;
          letter-spacing: -1px; margin-bottom: 10px;
        }
        .step-sub { color: var(--text-muted); font-size: 15px; margin-bottom: 32px; }
        .jd-input-wrap { margin-bottom: 28px; }

        .generate-btn {
          width: 100%; display: flex; align-items: center; justify-content: center;
          gap: 12px; padding: 18px 32px;
          background: var(--accent); color: #0c0e14;
          border: none; border-radius: var(--radius-lg);
          font-family: var(--font-body); font-size: 16px; font-weight: 600;
          cursor: pointer; transition: all 0.25s;
          letter-spacing: -0.2px;
        }
        .generate-btn:hover:not(:disabled) {
          background: #d4b47a; transform: translateY(-1px);
          box-shadow: 0 8px 32px var(--accent-glow);
        }
        .generate-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-arrow { font-size: 18px; transition: transform 0.2s; }
        .generate-btn:hover:not(:disabled) .btn-arrow { transform: translateX(4px); }

        /* Processing */
        .processing-section { align-items: center; }
        .processing-container { text-align: center; }
        .processing-pulse {
          position: relative; width: 80px; height: 80px;
          margin: 0 auto 36px; display: flex; align-items: center; justify-content: center;
        }
        .pulse-ring {
          position: absolute; inset: 0; border-radius: 50%;
          border: 2px solid var(--accent);
          animation: pulse-out 2s ease-out infinite;
        }
        .pulse-core {
          font-size: 28px; color: var(--accent); animation: pulse-in 2s ease-in-out infinite;
        }
        @keyframes pulse-out {
          0% { transform: scale(1); opacity: 1; }
          100% { transform: scale(2); opacity: 0; }
        }
        @keyframes pulse-in {
          0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
        }
        .processing-title {
          font-family: var(--font-display); font-size: 36px;
          letter-spacing: -1px; margin-bottom: 10px;
        }
        .processing-sub { color: var(--text-muted); margin-bottom: 40px; }
        .progress-wrap { max-width: 380px; margin: 0 auto; }
        .error-msg {
          margin-top: 24px; color: var(--red); font-size: 14px;
          background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.2);
          padding: 12px 16px; border-radius: var(--radius);
        }
        .error-actions {
          display: flex; gap: 12px; justify-content: center; margin-top: 16px;
        }
        .retry-btn {
          background: var(--accent); color: #0c0e14; border: none;
          padding: 10px 22px; border-radius: var(--radius);
          font-family: var(--font-body); font-size: 14px; font-weight: 600;
          cursor: pointer; transition: all 0.2s;
        }
        .retry-btn:hover { background: #d4b47a; }

        /* Results */
        .results-header {
          display: flex; align-items: flex-start; justify-content: space-between;
          gap: 24px; margin-bottom: 40px; flex-wrap: wrap;
        }
        .results-title {
          font-family: var(--font-display); font-size: 40px; letter-spacing: -1px; margin-bottom: 6px;
        }
        .results-sub { color: var(--text-muted); font-size: 15px; }
        .results-grid { display: grid; gap: 24px; }
        .result-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius-lg); overflow: hidden; }

        /* CVUploader override */
        .cv-uploader-override {
          background: var(--bg-2) !important;
          border-color: var(--border) !important;
        }
      `}</style>
    </div>
  );
}