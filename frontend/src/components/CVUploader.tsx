import { useRef, useState } from "react";

export function CVUploader({
  onFileSelected,
  maxSizeMb = 10,
}: {
  onFileSelected: (file: File) => void;
  maxSizeMb?: number;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function validate(file: File) {
    const name = file.name.toLowerCase();
    const okType = name.endsWith(".pdf") || name.endsWith(".docx");
    if (!okType) return "Only PDF or DOCX files are supported.";
    const okSize = file.size <= maxSizeMb * 1024 * 1024;
    if (!okSize) return `File too large. Max ${maxSizeMb}MB.`;
    return null;
  }

  function handleFile(file: File) {
    const err = validate(file);
    if (err) { setError(err); return; }
    setError(null);
    onFileSelected(file);
  }

  return (
    <div>
      <div
        style={{
          border: `1.5px dashed ${isDragging ? "var(--accent)" : "var(--border-bright)"}`,
          borderRadius: "var(--radius-lg)",
          padding: "48px 32px",
          textAlign: "center",
          background: isDragging ? "var(--accent-dim)" : "var(--bg-2)",
          cursor: "pointer",
          transition: "all 0.2s",
        }}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) handleFile(f);
        }}
        onClick={() => inputRef.current?.click()}
      >
        <div style={{ fontSize: 36, marginBottom: 16 }}>
          {isDragging ? "⬇" : "◈"}
        </div>
        <p style={{
          fontSize: 18, fontWeight: 600, marginBottom: 8,
          fontFamily: "var(--font-display)", letterSpacing: "-0.3px"
        }}>
          Drop your CV here
        </p>
        <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 24, lineHeight: 1.5 }}>
          PDF or DOCX · up to {maxSizeMb}MB
        </p>
        <button
          type="button"
          style={{
            background: "var(--accent)",
            color: "#0c0e14",
            border: "none",
            borderRadius: "10px",
            padding: "12px 28px",
            fontSize: 14,
            fontWeight: 600,
            cursor: "pointer",
            fontFamily: "var(--font-body)",
            transition: "all 0.2s",
          }}
          onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
        >
          Choose file
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
      </div>
      {error && (
        <p style={{
          marginTop: 12, fontSize: 13, color: "var(--red)",
          background: "rgba(248,113,113,0.1)",
          border: "1px solid rgba(248,113,113,0.2)",
          padding: "10px 14px", borderRadius: 8,
        }}>{error}</p>
      )}
    </div>
  );
}