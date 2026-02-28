import { useState } from "react";

type Props = {
  applicationId: string;
  accessToken: string;
  apiBaseUrl: string;
};

type DownloadTarget = {
  path: string;
  filename: string;
  label: string;
  primary?: boolean;
};

const DOWNLOADS: DownloadTarget[] = [
  { path: "enhanced-cv.docx", filename: "enhanced_cv.docx", label: "↓ Enhanced CV", primary: true },
  { path: "cover-letter.docx", filename: "cover_letter.docx", label: "↓ Cover Letter" },
];

export default function DownloadButton({ applicationId, accessToken, apiBaseUrl }: Props) {
  const [active, setActive] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const base = `${apiBaseUrl}/v1/applications/${applicationId}/download`;

  async function handleDownload(target: DownloadTarget) {
    setError(null);
    setActive(target.path);

    try {
      const res = await fetch(`${base}/${target.path}`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `Download failed (${res.status})`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = target.filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e?.message ?? "Download failed");
    } finally {
      setActive(null);
    }
  }

  const btnBase: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    padding: "10px 18px",
    borderRadius: 10,
    fontSize: 13,
    fontWeight: 500,
    border: "1px solid transparent",
    fontFamily: "var(--font-body)",
    transition: "all 0.2s",
    cursor: "pointer",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-end" }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
        {DOWNLOADS.map((target) => (
          <button
            key={target.path}
            type="button"
            onClick={() => handleDownload(target)}
            disabled={active !== null}
            style={
              target.primary
                ? { ...btnBase, background: "var(--accent)", color: "#0c0e14" }
                : { ...btnBase, background: "var(--bg-3)", color: "var(--text)", borderColor: "var(--border)" }
            }
          >
            {active === target.path ? "Preparing..." : target.label}
          </button>
        ))}
      </div>
      {error && <div style={{ fontSize: 12, color: "var(--red)", maxWidth: 320 }}>{error}</div>}
    </div>
  );
}
