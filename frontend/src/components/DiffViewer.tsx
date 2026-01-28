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
    <div className="rounded-2xl border p-4 shadow-sm">
      <div className="flex items-center justify-between gap-4">
        <h2 className="text-xl font-semibold">CV Suggestions</h2>
        <div className="text-sm opacity-80">
          Acceptance rate: {Math.round(acceptanceRate * 100)}%
        </div>
      </div>

      <div className="mt-4 grid gap-3">
        {suggestions.length === 0 ? (
          <p className="text-sm opacity-70">No suggestions available.</p>
        ) : (
          suggestions.map((s) => (
            <div key={s.id} className="rounded-xl border p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="font-medium">{s.section}</div>

                <div className="flex gap-2">
                  <button
                    className="rounded-lg border px-3 py-1 text-sm hover:opacity-80"
                    onClick={() => setDecision(s.id, "accepted")}
                  >
                    Accept
                  </button>
                  <button
                    className="rounded-lg border px-3 py-1 text-sm hover:opacity-80"
                    onClick={() => setDecision(s.id, "rejected")}
                  >
                    Reject
                  </button>
                </div>
              </div>

              <div className="mt-3 grid gap-2 md:grid-cols-2">
                <div>
                  <div className="text-xs font-semibold opacity-70">Before</div>
                  <div className="mt-1 whitespace-pre-wrap rounded-lg border p-2 text-sm">
                    {s.before}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold opacity-70">After</div>
                  <div className="mt-1 whitespace-pre-wrap rounded-lg border p-2 text-sm">
                    {s.after}
                  </div>
                </div>
              </div>

              {decisions[s.id] ? (
                <div className="mt-2 text-xs opacity-70">Decision: {decisions[s.id]}</div>
              ) : null}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
