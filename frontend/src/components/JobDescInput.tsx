export function JobDescInput({
  value,
  onChange,
  maxChars = 8000,
}: {
  value: string;
  onChange: (v: string) => void;
  maxChars?: number;
}) {
  const words = value.trim() ? value.trim().split(/\\s+/).length : 0;

  return (
    <div>
      <label className="block text-sm font-medium">Job description</label>
      <textarea
        className="mt-2 w-full rounded-2xl border p-4 text-sm outline-none focus:ring-2"
        rows={10}
        value={value}
        onChange={(e) => onChange(e.target.value.slice(0, maxChars))}
        placeholder="Paste the job description here…"
      />
      <div className="mt-2 flex items-center justify-between text-xs text-gray-600">
        <span>{words} words</span>
        <span>
          {value.length}/{maxChars} chars
        </span>
      </div>
    </div>
  );
}

