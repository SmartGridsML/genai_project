import axios, { AxiosError } from "axios";

export type ApiError = {
  message: string;
  status?: number;
  detail?: unknown;
};

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 60_000,
});

api.interceptors.response.use(
  (resp) => resp,
  (err: AxiosError) => {
    const status = err.response?.status;
    const data = err.response?.data as any;
    const apiErr: ApiError = {
      message: data?.message ?? data?.detail ?? err.message ?? "Request failed",
      status,
      detail: data,
    };
    return Promise.reject(apiErr);
  }
);

