import type { AuditClaim } from "../types/application";

type Props = {
  coverLetter: string;
  audit: AuditClaim[];
};

function highlightClaims(text: string, audit: AuditClaim[]) {
  const claims = [...audit]
    .filter((c) => c.claim?.trim())
    .sort((a, b) => b.claim.length - a.claim.length);

  let parts: Array<{ text: string; tag?: "supported" | "flagged" }> = [{ text }];

  for (const c of claims) {
    const next: typeof parts = [];
    for (const p of parts) {
      if (p.tag) { next.push(p); continue; }
      const idx = p.text.toLowerCase().indexOf(c.claim.toLowerCase());
      if (idx === -1) { next.push(p); continue; }
      if (p.text.slice(0, idx)) next.push({ text: p.text.slice(0, idx) });
      next.push({ text: p.text.slice(idx, idx + c.claim.length), tag: c.supported ? "supported" : "flagged" });
      if (p.text.slice(idx + c.claim.length)) next.push({ text: p.text.slice(idx + c.claim.length) });
    }
    parts = next;
  }
  return parts;
}

export default function ResultsViewer({ coverLetter, audit }: Props) {
  const parts = highlightClaims(coverLetter, audit);
  const supportedCount = audit.filter(a => a.supported).length;
  const flaggedCount = audit.filter(a => !a.supported).length;

  return (
    <div style={{ padding: 28 }}>
      {/* Cover Letter */}
      <div style={{ marginBottom: 32 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
          <h2 style={{ fontFamily: "var(--font-display)", fontSize: 26, letterSpacing: "-0.5px" }}>
            Cover Letter
          </h2>
          <div style={{ display: "flex", gap: 8 }}>
            {supportedCount > 0 && (
              <span style={{
                fontSize: 12, padding: "4px 10px", borderRadius: 999,
                background: "rgba(74,222,128,0.1)", color: "var(--green)",
                border: "1px solid rgba(74,222,128,0.2)",
              }}>
                {supportedCount} verified
              </span>
            )}
            {flaggedCount > 0 && (
              <span style={{
                fontSize: 12, padding: "4px 10px", borderRadius: 999,
                background: "rgba(248,113,113,0.1)", color: "var(--red)",
                border: "1px solid rgba(248,113,113,0.2)",
              }}>
                {flaggedCount} flagged
              </span>
            )}
          </div>
        </div>

        <div style={{
          background: "var(--bg-3)", borderRadius: "var(--radius)",
          padding: "24px 28px", lineHeight: 1.9, fontSize: 15,
          whiteSpace: "pre-wrap", color: "var(--text)",
          border: "1px solid var(--border)",
        }}>
          {parts.map((p, i) => {
            if (!p.tag) return <span key={i}>{p.text}</span>;
            return (
              <span
                key={i}
                title={p.tag === "supported" ? "✓ Supported claim" : "⚠ Flagged claim"}
                style={{
                  background: p.tag === "supported"
                    ? "rgba(74,222,128,0.12)"
                    : "rgba(248,113,113,0.12)",
                  borderBottom: `2px solid ${p.tag === "supported" ? "var(--green)" : "var(--red)"}`,
                  padding: "1px 2px",
                  borderRadius: 3,
                  cursor: "help",
                }}
              >
                {p.text}
              </span>
            );
          })}
        </div>

        {audit.length > 0 && (
          <p style={{ marginTop: 10, fontSize: 12, color: "var(--text-muted)" }}>
            Underlined claims: <span style={{ color: "var(--green)" }}>green = verified</span> · <span style={{ color: "var(--red)" }}>red = flagged</span>
          </p>
        )}
      </div>

      {/* Audit Report */}
      <div>
        <h2 style={{ fontFamily: "var(--font-display)", fontSize: 26, letterSpacing: "-0.5px", marginBottom: 20 }}>
          Audit Report
        </h2>
        {audit.length === 0 ? (
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>No claims detected.</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {audit.map((c, idx) => (
              <div key={idx} style={{
                display: "flex", alignItems: "flex-start", gap: 14,
                background: "var(--bg-3)", border: "1px solid var(--border)",
                borderRadius: "var(--radius)", padding: "14px 16px",
                borderLeft: `3px solid ${c.supported ? "var(--green)" : "var(--red)"}`,
              }}>
                <div style={{ fontSize: 16, marginTop: 1 }}>
                  {c.supported ? "✓" : "⚠"}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 4 }}>{c.claim}</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    <span style={{ color: c.supported ? "var(--green)" : "var(--red)" }}>
                      {c.supported ? "Supported" : "Flagged"}
                    </span>
                    {typeof c.confidence === "number" && (
                      <span> · {Math.round(c.confidence * 100)}% confidence</span>
                    )}
                    {c.source && <span> · {c.source}</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}