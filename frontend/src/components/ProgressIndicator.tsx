export type Stage =
  | "idle"
  | "parsing_cv"
  | "generating_letter"
  | "auditing"
  | "done"
  | "failed";

const STAGES: Array<{ key: Exclude<Stage, "idle" | "done" | "failed">; label: string }> = [
  { key: "parsing_cv", label: "Parsing CV" },
  { key: "generating_letter", label: "Generating application" },
  { key: "auditing", label: "Auditing claims" },
];

export function ProgressIndicator({ stage }: { stage: Stage }) {
  const activeIndex = STAGES.findIndex((s) => s.key === stage);

  return (
    <div style={{
      background: "var(--bg-2)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius-lg)",
      padding: "24px 28px",
    }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        {STAGES.map((s, idx) => {
          const done = stage === "done" || activeIndex > idx;
          const active = activeIndex === idx;
          const isLast = idx === STAGES.length - 1;
          return (
            <div key={s.key} style={{ display: "flex", gap: 16 }}>
              {/* Timeline */}
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 20 }}>
                <div style={{
                  width: 20, height: 20, borderRadius: "50%", flexShrink: 0,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  background: done ? "var(--accent)" : active ? "transparent" : "transparent",
                  border: done ? "none" : active ? "2px solid var(--accent)" : "2px solid var(--border)",
                  transition: "all 0.3s",
                  fontSize: 10,
                }}>
                  {done && <span style={{ color: "#0c0e14", fontWeight: 700 }}>✓</span>}
                  {active && (
                    <span style={{
                      width: 8, height: 8, borderRadius: "50%",
                      background: "var(--accent)",
                      display: "block",
                      animation: "blink 1.2s ease-in-out infinite",
                    }} />
                  )}
                </div>
                {!isLast && (
                  <div style={{
                    width: 2, flex: 1, minHeight: 20,
                    background: done ? "var(--accent)" : "var(--border)",
                    transition: "background 0.3s",
                    margin: "3px 0",
                  }} />
                )}
              </div>
              {/* Label */}
              <div style={{ paddingBottom: isLast ? 0 : 20, paddingTop: 1 }}>
                <span style={{
                  fontSize: 14,
                  color: done ? "var(--text)" : active ? "var(--accent)" : "var(--text-muted)",
                  fontWeight: active ? 500 : 400,
                  transition: "color 0.3s",
                }}>
                  {s.label}
                  {active && <span style={{ marginLeft: 8, fontSize: 12, opacity: 0.6 }}>...</span>}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {(stage === "done" || stage === "failed") && (
        <div style={{
          marginTop: 16, padding: "10px 14px",
          borderRadius: 8, fontSize: 13, fontWeight: 500,
          background: stage === "done" ? "rgba(74,222,128,0.1)" : "rgba(248,113,113,0.1)",
          color: stage === "done" ? "var(--green)" : "var(--red)",
          border: `1px solid ${stage === "done" ? "rgba(74,222,128,0.2)" : "rgba(248,113,113,0.2)"}`,
        }}>
          {stage === "done" ? "✓ Complete" : "✕ Failed"}
        </div>
      )}

      <style>{`
        @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.2; } }
      `}</style>
    </div>
  );
}