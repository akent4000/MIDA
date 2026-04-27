import { OpenAPI } from "@/api";

OpenAPI.BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export function apiUrl(path: string): string {
  const base = OpenAPI.BASE.replace(/\/$/, "");
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

export function wsUrl(path: string): string {
  const base = import.meta.env.VITE_WS_BASE_URL;
  if (base) return `${base.replace(/\/$/, "")}${path}`;
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}
