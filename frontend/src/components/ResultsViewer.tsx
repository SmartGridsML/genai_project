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
      if (p.tag) {
        next.push(p);
        continue;
      }

      const hay = p.text;
      const needle = c.claim;

      const idx = hay.toLowerCase().indexOf(needle.toLowerCase());
      if (idx === -1) {
        next.push(p);
        continue;
      }

      const before = hay.slice(0, idx);
      const match = hay.slice(idx, idx + needle.length);
      const after = hay.slice(idx + needle.length);

      if (before) next.push({ text: before });
      next.push({ text: match, tag: c.supported ? "supported" : "flagged" });
      if (after) next.push({ text: after });
    }
    parts = next;
  }

  return parts;
}

export default function ResultsViewer({ coverLetter, audit }: Props) {
  const parts = highlightClaims(coverLetter, audit);

  return (
    <div className="grid gap-6">
      <div className="rounded-2xl border p-4 shadow-sm">
        <h2 className="text-xl font-semibold">Cover Letter</h2>
        <div className="mt-3 whitespace-pre-wrap leading-7">
          {parts.map((p, i) => {
            if (!p.tag) return <span key={i}>{p.text}</span>;

            const cls =
              p.tag === "supported"
                ? "rounded px-1 underline decoration-2"
                : "rounded px-1 underline decoration-2";

            return (
              <span
                key={i}
                className={cls}
                title={p.tag === "supported" ? "Supported claim" : "Flagged claim"}
              >
                {p.text}
              </span>
            );
          })}
        </div>
      </div>

      <div className="rounded-2xl border p-4 shadow-sm">
        <h2 className="text-xl font-semibold">Audit Report</h2>

        <div className="mt-3 grid gap-2">
          {audit.length === 0 ? (
            <p className="text-sm opacity-70">No claims detected.</p>
          ) : (
            audit.map((c, idx) => (
              <div key={idx} className="flex items-start gap-3 rounded-xl border p-3">
                <div className="mt-0.5 text-lg">{c.supported ? "✅" : "⚠️"}</div>

                <div className="min-w-0">
                  <div className="font-medium">{c.claim}</div>

                  <div className="mt-1 text-sm opacity-80">
                    {c.supported ? "Supported" : "Flagged"}
                    {typeof c.confidence === "number"
                      ? ` • confidence: ${Math.round(c.confidence * 100)}%`
                      : ""}
                  </div>

                  {c.source ? (
                    <div className="mt-1 text-xs opacity-70">Source: {c.source}</div>
                  ) : null}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
