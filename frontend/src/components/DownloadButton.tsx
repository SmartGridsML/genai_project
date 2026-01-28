type Props = {
  applicationId: string;
  apiBaseUrl: string;
};

export default function DownloadButton({ applicationId, apiBaseUrl }: Props) {
  const cvUrl = `${apiBaseUrl}/applications/${applicationId}/download?type=cv`;
  const clUrl = `${apiBaseUrl}/applications/${applicationId}/download?type=cover_letter`;
  const zipUrl = `${apiBaseUrl}/applications/${applicationId}/download?type=zip`;

  return (
    <div className="flex flex-wrap gap-2">
      <a className="rounded-xl border px-4 py-2 text-sm hover:opacity-80" href={cvUrl}>
        Download enhanced CV
      </a>
      <a className="rounded-xl border px-4 py-2 text-sm hover:opacity-80" href={clUrl}>
        Download cover letter
      </a>
      <a className="rounded-xl border px-4 py-2 text-sm hover:opacity-80" href={zipUrl}>
        Download ZIP
      </a>
    </div>
  );
}
