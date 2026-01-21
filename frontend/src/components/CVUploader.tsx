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
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    onFileSelected(file);
  }

  return (
    <div
      className={[
        "rounded-2xl border border-dashed p-6 text-center transition",
        isDragging ? "border-black bg-gray-50" : "border-gray-300 bg-white",
      ].join(" ")}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f) handleFile(f);
      }}
    >
      <p className="text-lg font-semibold">Upload your CV</p>
      <p className="mt-1 text-sm text-gray-600">Drag & drop a PDF/DOCX, or click to choose</p>

      <button
        type="button"
        className="mt-4 rounded-xl border px-4 py-2 text-sm font-medium hover:bg-gray-50"
        onClick={() => inputRef.current?.click()}
      >
        Choose file
      </button>

      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />

      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
    </div>
  );
}

