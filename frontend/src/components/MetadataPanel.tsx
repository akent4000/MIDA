import type { StudyPublic } from "@/api";

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

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value || "—";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map(formatValue).join(", ");
  return JSON.stringify(value);
}

interface Row {
  label: string;
  value: string;
}

function FileSection({ study }: { study: StudyPublic }) {
  const rows: Row[] = [
    { label: "Формат", value: study.file_format.toUpperCase() },
    { label: "Размер", value: formatBytes(study.file_size) },
    { label: "Анонимизирован", value: study.anonymized ? "да" : "нет" },
    { label: "Загружен", value: formatDate(study.created_at) },
  ];
  return <Section title="Файл" rows={rows} />;
}

function Section({ title, rows }: { title: string; rows: Row[] }) {
  if (rows.length === 0) return null;
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </h3>
      <dl className="space-y-1.5 text-sm">
        {rows.map((row) => (
          <div key={row.label} className="grid grid-cols-[1fr_1.5fr] gap-3">
            <dt className="text-muted-foreground">{row.label}</dt>
            <dd className="break-words font-mono text-xs text-foreground">{row.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function MetadataPanel({ study }: { study: StudyPublic }) {
  const meta = study.dicom_metadata ?? {};
  const metaRows: Row[] = Object.keys(meta)
    .sort()
    .map((key) => ({ label: key, value: formatValue(meta[key]) }));

  return (
    <div className="space-y-5">
      <FileSection study={study} />
      {metaRows.length > 0 ? (
        <Section title="DICOM-метаданные" rows={metaRows} />
      ) : (
        <p className="text-xs text-muted-foreground">Метаданные DICOM отсутствуют.</p>
      )}
    </div>
  );
}
