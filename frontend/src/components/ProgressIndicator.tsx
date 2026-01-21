export type Stage =
  | "idle"
  | "parsing_cv"
  | "extracting_facts"
  | "analyzing_jd"
  | "generating_letter"
  | "auditing"
  | "done"
  | "failed";

const STAGES: Array<{ key: Exclude<Stage, "idle" | "done" | "failed">; label: string }> = [
  { key: "parsing_cv", label: "Parsing CV" },
  { key: "extracting_facts", label: "Extracting facts" },
  { key: "analyzing_jd", label: "Analyzing job description" },
  { key: "generating_letter", label: "Generating cover letter" },
  { key: "auditing", label: "Auditing" },
];

export function ProgressIndicator({ stage }: { stage: Stage }) {
  const activeIndex = STAGES.findIndex((s) => s.key === stage);

  return (
    <div className="rounded-2xl border p-4">
      <p className="text-sm font-medium">Progress</p>
      <div className="mt-3 space-y-2">
        {STAGES.map((s, idx) => {
          const done = stage === "done" || activeIndex > idx;
          const active = activeIndex === idx;
          return (
            <div key={s.key} className="flex items-center gap-3">
              <div
                className={[
                  "h-3 w-3 rounded-full border",
                  done ? "bg-black" : active ? "bg-gray-400" : "bg-white",
                ].join(" ")}
              />
              <span className={["text-sm", done ? "text-black" : "text-gray-700"].join(" ")}>
                {s.label}
              </span>
            </div>
          );
        })}
        {stage === "failed" && <p className="text-sm text-red-600">Failed</p>}
        {stage === "done" && <p className="text-sm text-green-700">Done</p>}
      </div>
    </div>
  );
}

