import { Link } from "react-router-dom";
import { Loader2, RefreshCw } from "lucide-react";
import type { StudyPublic } from "@/api";
import { Button } from "@/components/ui/button";
import { useStudiesList } from "@/hooks/useStudies";
import { formatApiError } from "@/lib/errors";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString("ru-RU");
  } catch {
    return iso;
  }
}

function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
}

export function HistoryPage() {
  const { data, isLoading, isError, error, isFetching, refetch } = useStudiesList();

  return (
    <section className="space-y-4">
      <header className="flex items-end justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">История исследований</h1>
          <p className="text-sm text-muted-foreground">
            Список ранее загруженных исследований.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
          aria-label="Обновить"
        >
          {isFetching ? (
            <Loader2 className="mr-1 h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <RefreshCw className="mr-1 h-4 w-4" aria-hidden />
          )}
          Обновить
        </Button>
      </header>

      {isLoading && (
        <p className="text-sm text-muted-foreground">Загрузка истории…</p>
      )}

      {isError && (
        <p
          role="alert"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
        >
          {formatApiError(error)}
        </p>
      )}

      {data && data.length === 0 && !isLoading && (
        <div className="rounded-lg border border-border bg-card/40 p-6 text-sm text-muted-foreground">
          История пуста. Загрузите первое исследование на странице{" "}
          <Link to="/" className="text-primary underline-offset-4 hover:underline">
            «Загрузка»
          </Link>
          .
        </div>
      )}

      {data && data.length > 0 && <StudiesTable studies={data} />}
    </section>
  );
}

function StudiesTable({ studies }: { studies: StudyPublic[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card/40">
      <table className="w-full text-sm">
        <thead className="bg-secondary/40 text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th className="px-3 py-2 text-left font-medium">ID</th>
            <th className="px-3 py-2 text-left font-medium">Формат</th>
            <th className="px-3 py-2 text-right font-medium">Размер</th>
            <th className="px-3 py-2 text-left font-medium">Загружен</th>
            <th className="px-3 py-2 text-right font-medium" />
          </tr>
        </thead>
        <tbody>
          {studies.map((s) => (
            <tr
              key={s.id}
              className="border-t border-border transition-colors hover:bg-secondary/20"
            >
              <td className="px-3 py-2 font-mono text-xs">{shortId(s.id)}</td>
              <td className="px-3 py-2 uppercase">{s.file_format}</td>
              <td className="px-3 py-2 text-right font-mono tabular-nums">
                {formatBytes(s.file_size)}
              </td>
              <td className="px-3 py-2 text-muted-foreground">{formatDate(s.created_at)}</td>
              <td className="px-3 py-2 text-right">
                <Link
                  to={`/studies/${s.id}`}
                  className="text-primary underline-offset-4 hover:underline"
                >
                  Открыть
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
