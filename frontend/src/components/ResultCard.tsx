import type { InferenceResultPublic } from "@/api";

interface RawResult {
  tool_id?: string;
  prob?: number | null;
  label?: number | null;
  label_name?: string | null;
  threshold?: number | null;
  class_names?: string[] | null;
}

interface PostprocessedResult {
  interpretation?: string;
  confidence_band?: string;
  extra?: Record<string, unknown> | null;
  raw?: RawResult;
}

const BAND_LABEL: Record<string, string> = {
  low: "низкая",
  medium: "средняя",
  high: "высокая",
};

const BAND_TONE: Record<string, string> = {
  low: "border-yellow-500/40 bg-yellow-500/10 text-yellow-200",
  medium: "border-blue-500/40 bg-blue-500/10 text-blue-200",
  high: "border-emerald-500/40 bg-emerald-500/10 text-emerald-200",
};

function formatProb(prob: number | null | undefined): string {
  if (prob === null || prob === undefined || Number.isNaN(prob)) return "—";
  return `${(prob * 100).toFixed(1)}%`;
}

export function ResultCard({ result }: { result: InferenceResultPublic }) {
  const data = (result.result ?? {}) as PostprocessedResult;
  const raw = data.raw ?? {};
  const band = (data.confidence_band ?? "").toLowerCase();
  const bandTone = BAND_TONE[band] ?? "border-border bg-secondary/40 text-foreground";
  const bandLabel = BAND_LABEL[band] ?? (band || "—");

  return (
    <div className="space-y-3 rounded-md border border-border bg-card/40 p-3 text-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Результат
        </h3>
        <span className="font-mono text-[10px] text-muted-foreground">{result.tool_id}</span>
      </div>

      <div>
        <div className="text-xs text-muted-foreground">Заключение</div>
        <p className="font-medium">{data.interpretation ?? raw.label_name ?? "—"}</p>
      </div>

      <dl className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <dt className="text-muted-foreground">Вероятность</dt>
          <dd className="font-mono tabular-nums">{formatProb(raw.prob)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Порог</dt>
          <dd className="font-mono tabular-nums">{formatProb(raw.threshold)}</dd>
        </div>
      </dl>

      {band && (
        <div className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] ${bandTone}`}>
          Уверенность: {bandLabel}
        </div>
      )}
    </div>
  );
}
