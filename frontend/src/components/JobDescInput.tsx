export function JobDescInput({
  value,
  onChange,
  maxChars = 8000,
}: {
  value: string;
  onChange: (v: string) => void;
  maxChars?: number;
}) {
  const words = value.trim() ? value.trim().split(/\s+/).length : 0;
  const pct = value.length / maxChars;

  return (
    <div>
      <textarea
        style={{
          width: "100%",
          minHeight: 280,
          background: "var(--bg-2)",
          border: "1.5px solid var(--border)",
          borderRadius: "var(--radius-lg)",
          padding: "20px 24px",
          fontSize: 14,
          lineHeight: 1.7,
          color: "var(--text)",
          fontFamily: "var(--font-body)",
          resize: "vertical",
          outline: "none",
          transition: "border-color 0.2s",
          caretColor: "var(--accent)",
        }}
        value={value}
        onChange={(e) => onChange(e.target.value.slice(0, maxChars))}
        placeholder="Paste the job description here…"
        onFocus={(e) => { e.target.style.borderColor = "var(--accent)"; }}
        onBlur={(e) => { e.target.style.borderColor = "var(--border)"; }}
      />
      <div style={{
        marginTop: 10, display: "flex",
        alignItems: "center", justifyContent: "space-between",
        fontSize: 12, color: "var(--text-muted)",
      }}>
        <span>{words} words</span>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 80, height: 3,
            background: "var(--bg-3)", borderRadius: 2, overflow: "hidden",
          }}>
            <div style={{
              width: `${pct * 100}%`, height: "100%",
              background: pct > 0.9 ? "var(--red)" : "var(--accent)",
              borderRadius: 2, transition: "width 0.2s",
            }} />
          </div>
          <span>{value.length}/{maxChars}</span>
        </div>
      </div>
    </div>
  );
}
