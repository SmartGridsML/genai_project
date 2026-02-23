import { useMemo, useState } from "react";
import { CVUploader } from "../components/CVUploader";
import { JobDescInput } from "../components/JobDescInput";
import { ProgressIndicator } from "../components/ProgressIndicator";
import type { Stage } from "../components/ProgressIndicator";
import { generateApplication, parseCv, pollResults } from "../api/applications";
import type { ResultsResponse } from "../api/applications";
import type { ApiError } from "../api/client";
import ResultsViewer from "../components/ResultsViewer";
import DiffViewer from "../components/DiffViewer";
import DownloadButton from "../components/DownloadButton";


export function Home() {
  const [cvText, setCvText] = useState("");
  const [jd, setJd] = useState("");
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [applicationId, setApplicationId] = useState<string | null>(null);

  const canSubmit = useMemo(() => cvText.trim().length > 0 && jd.trim().length > 0, [cvText, jd]);

  async function onUpload(file: File) {
    setError(null);
    setResults(null);
    setStage("parsing_cv");
    setApplicationId(null);
    try {
      const parsed = await parseCv(file);
      setCvText(parsed.parsed_text ?? "");
      setStage("idle");
    } catch (e) {
      setStage("failed");
      setError((e as ApiError).message);
    }
  }

  async function onGenerate() {
    setError(null);
    setResults(null);
    setApplicationId(null);
    try {
      setStage("extracting_facts");
      setStage("analyzing_jd");
      setStage("generating_letter");

      const gen = await generateApplication({ cv_text: cvText, job_description: jd });
      setApplicationId(gen.application_id);

      setStage("auditing");
      const res = await pollResults(gen.application_id);
      setResults(res);

      setStage(res.status === "done" ? "done" : "failed");
      if (res.status === "failed") setError(res.error ?? "Generation failed");
    } catch (e) {
      setStage("failed");
      setError((e as ApiError).message);
    }
  }

  return (
    <div className="mx-auto max-w-5xl p-6">
      <header className="mb-8">
        <h1 className="text-2xl font-bold">Cover Letter Generator</h1>
        <p className="mt-1 text-gray-600">Upload CV + paste JD → generate grounded cover letter.</p>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-6">
          <CVUploader onFileSelected={onUpload} />

          <div className="rounded-2xl border p-4">
            <p className="text-sm font-medium">Extracted CV text</p>
            <textarea
              className="mt-3 w-full rounded-xl border p-3 text-xs outline-none focus:ring-2"
              rows={10}
              value={cvText}
              onChange={(e) => setCvText(e.target.value)}
              placeholder="CV text will appear here after parsing…"
            />
          </div>
        </div>

        <div className="space-y-6">
          <JobDescInput value={jd} onChange={setJd} />
          <ProgressIndicator stage={stage} />

          <button
            className={[
              "w-full rounded-2xl px-4 py-3 text-sm font-semibold",
              canSubmit && stage !== "auditing" && stage !== "generating_letter"
                ? "bg-black text-white hover:opacity-90"
                : "cursor-not-allowed bg-gray-200 text-gray-600",
            ].join(" ")}
            disabled={!canSubmit || stage === "auditing" || stage === "generating_letter"}
            onClick={onGenerate}
          >
            Generate
          </button>

          {error && (
            <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
              {error}
            </div>
          )}
        </div>
      </div>

      {results?.status === "done" && (
        <div className="mt-8 grid gap-6">
          {applicationId && (
            <DownloadButton
              applicationId={applicationId}
              apiBaseUrl={import.meta.env.VITE_API_BASE_URL}
            />
          )}

          <ResultsViewer
            coverLetter={results.cover_letter ?? ""}
            audit={(results.audit_report ?? []).map((a) => ({
              claim: a.claim,
              supported: a.supported,
              source: a.source,
              confidence: a.confidence,
            }))}
          />

          {/* Enable once backend provides suggestions */}
          <DiffViewer suggestions={[]} />
        </div>
      )}
    </div>
  );
}

