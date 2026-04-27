import { ApiError } from "@/api";

export function formatApiError(err: unknown): string {
  if (err instanceof ApiError) {
    const detail = err.body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      return detail.map((d) => d?.msg ?? JSON.stringify(d)).join("; ");
    }
    return `${err.status} ${err.statusText}`;
  }
  if (err instanceof Error) return err.message;
  return "Неизвестная ошибка";
}
