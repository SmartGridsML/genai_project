type Props = {
  applicationId: string;
  apiBaseUrl: string;
};

export default function DownloadButton({ applicationId, apiBaseUrl }: Props) {
  const cvUrl = `${apiBaseUrl}/applications/${applicationId}/download?type=cv`;
  const clUrl = `${apiBaseUrl}/applications/${applicationId}/download?type=cover_letter`;
  const zipUrl = `${apiBaseUrl}/applications/${applicationId}/download?type=zip`;

  const btnBase: React.CSSProperties = {
    display: "inline-flex", alignItems: "center", gap: 8,
    padding: "10px 18px", borderRadius: 10,
    fontSize: 13, fontWeight: 500, textDecoration: "none",
    fontFamily: "var(--font-body)", transition: "all 0.2s",
    cursor: "pointer",
  };

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
      <a
        href={cvUrl}
        style={{ ...btnBase, background: "var(--accent)", color: "#0c0e14" }}
      >
        ↓ Enhanced CV
      </a>
      <a
        href={clUrl}
        style={{ ...btnBase, background: "var(--bg-3)", color: "var(--text)", border: "1px solid var(--border)" }}
      >
        ↓ Cover Letter
      </a>
      <a
        href={zipUrl}
        style={{ ...btnBase, background: "var(--bg-3)", color: "var(--text)", border: "1px solid var(--border)" }}
      >
        ↓ ZIP Bundle
      </a>
    </div>
  );
}
