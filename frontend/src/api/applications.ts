import { api } from "./client";

export type ParseResponse = {
  parsed_text: string;
  sections?: Record<string, string>;
};

export type GenerateRequest = {
  cv_text: string;
  job_description: string;
};

export type GenerateResponse = {
  application_id: string;
  status?: "queued" | "processing" | "done" | "failed";
};

export type AuditItem = {
  claim: string;
  supported: boolean;
  source: string;
  confidence: number;
};

export type ResultsResponse = {
  application_id: string;
  status: "processing" | "done" | "failed";
  cover_letter?: string;
  cv_suggestions?: Array<{ before: string; after: string }>;
  audit_report?: AuditItem[];
  error?: string;
};

export async function parseCv(file: File): Promise<ParseResponse> {
  const form = new FormData();
  form.append("file", file);

  const { data } = await api.post<ParseResponse>("/applications/parse", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function generateApplication(req: GenerateRequest): Promise<GenerateResponse> {
  const { data } = await api.post<GenerateResponse>("/applications/generate", req);
  return data;
}

export async function getResults(id: string): Promise<ResultsResponse> {
  const { data } = await api.get<ResultsResponse>(`/applications/${id}/results`);
  return data;
}

export async function pollResults(
  id: string,
  opts?: { intervalMs?: number; timeoutMs?: number }
): Promise<ResultsResponse> {
  const intervalMs = opts?.intervalMs ?? 1500;
  const timeoutMs = opts?.timeoutMs ?? 90_000;
  const start = Date.now();

  while (true) {
    const res = await getResults(id);
    if (res.status === "done" || res.status === "failed") return res;

    if (Date.now() - start > timeoutMs) {
      return { application_id: id, status: "failed", error: "Polling timed out" };
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

