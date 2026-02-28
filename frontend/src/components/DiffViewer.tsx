import { useMemo, useState } from "react";
import type { CvSuggestion } from "../types/application";

type Decision = "accepted" | "rejected";
type Props = {
  suggestions: CvSuggestion[];
  onChange?: (decisions: Record<string, Decision>) => void;
};

export default function DiffViewer({ suggestions, onChange }: Props) {
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});

  const acceptanceRate = useMemo(() => {
    const total = suggestions.length;
    if (total === 0) return 0;
    const accepted = Object.values(decisions).filter((d) => d === "accepted").length;
    return accepted / total;
  }, [decisions, suggestions.length]);

  function setDecision(id: string, d: Decision) {
    setDecisions((prev) => {
      const next = { ...prev, [id]: d };
      onChange?.(next);
      return next;
    });
  }

  return (
    <div style={{ padding: 28 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <h2 style={{ fontFamily: "var(--font-display)", fontSize: 26, letterSpacing: "-0.5px" }}>
          CV Suggestions
        </h2>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: "var(--accent)", fontFamily: "var(--font-display)" }}>
            {Math.round(acceptanceRate * 100)}%
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.5px" }}>ACCEPTED</div>
        </div>
      </div>

      {suggestions.length === 0 ? (
        <p style={{ color: "var(--text-muted)", fontSize: 14 }}>No suggestions available.</p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {suggestions.map((s) => {
            const dec = decisions[s.id];
            return (
              <div key={s.id} style={{
                background: "var(--bg-3)", border: "1px solid var(--border)",
                borderRadius: "var(--radius)", overflow: "hidden",
                opacity: dec === "rejected" ? 0.5 : 1, transition: "opacity 0.2s",
                borderLeft: dec === "accepted" ? "3px solid var(--green)"
                  : dec === "rejected" ? "3px solid var(--red)"
                  : "3px solid var(--border)",
              }}>
                <div style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "12px 16px",
                  borderBottom: "1px solid var(--border)",
                }}>
                  <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-muted)" }}>
                    {s.section}
                  </span>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button
                      style={{
                        background: dec === "accepted" ? "var(--green)" : "transparent",
                        color: dec === "accepted" ? "#0c0e14" : "var(--green)",
                        border: "1px solid var(--green)",
                        borderRadius: 8, padding: "5px 14px",
                        fontSize: 12, fontWeight: 600,
                        cursor: "pointer", fontFamily: "var(--font-body)",
                        transition: "all 0.2s",
                      }}
                      onClick={() => setDecision(s.id, "accepted")}
                    >
                      Accept
                    </button>
                    <button
                      style={{
                        background: dec === "rejected" ? "var(--red)" : "transparent",
                        color: dec === "rejected" ? "#0c0e14" : "var(--red)",
                        border: "1px solid var(--red)",
                        borderRadius: 8, padding: "5px 14px",
                        fontSize: 12, fontWeight: 600,
                        cursor: "pointer", fontFamily: "var(--font-body)",
                        transition: "all 0.2s",
                      }}
                      onClick={() => setDecision(s.id, "rejected")}
                    >
                      Reject
                    </button>
                  </div>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
                  <div style={{ padding: "14px 16px", borderRight: "1px solid var(--border)" }}>
                    <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: 1, color: "var(--text-muted)", marginBottom: 8, textTransform: "uppercase" }}>Before</div>
                    <div style={{ fontSize: 13, lineHeight: 1.6, color: "var(--text-muted)", whiteSpace: "pre-wrap" }}>{s.before}</div>
                  </div>
                  <div style={{ padding: "14px 16px" }}>
                    <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: 1, color: "var(--accent)", marginBottom: 8, textTransform: "uppercase" }}>After</div>
                    <div style={{ fontSize: 13, lineHeight: 1.6, color: "var(--text)", whiteSpace: "pre-wrap" }}>{s.after}</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}