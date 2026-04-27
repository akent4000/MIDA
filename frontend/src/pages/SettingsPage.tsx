import { useState } from "react";
import { Loader2, Save } from "lucide-react";
import type { ModelListItem, SettingField, ToolSettingsConfig } from "@/api";
import { Button } from "@/components/ui/button";
import { useModels } from "@/hooks/useInference";
import { usePatchToolConfig, useToolConfig } from "@/hooks/useToolSettings";
import { formatApiError } from "@/lib/errors";
import { cn } from "@/lib/utils";

export function SettingsPage() {
  const models = useModels();

  return (
    <section className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Настройки моделей</h1>
        <p className="text-sm text-muted-foreground">
          Конфигурация для каждой ML-модели в реестре. Изменения применяются на следующем
          запуске инференса без перезапуска воркера.
        </p>
      </header>

      {models.isLoading && (
        <p className="text-sm text-muted-foreground">Загрузка списка моделей…</p>
      )}

      {models.isError && (
        <p
          role="alert"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
        >
          {formatApiError(models.error)}
        </p>
      )}

      {models.data && models.data.length === 0 && (
        <p className="text-sm text-muted-foreground">В реестре нет ни одной модели.</p>
      )}

      {models.data && models.data.length > 0 && (
        <div className="space-y-4">
          {models.data.map((m) => (
            <ToolCard key={m.tool_id} model={m} />
          ))}
        </div>
      )}
    </section>
  );
}

function ToolCard({ model }: { model: ModelListItem }) {
  const config = useToolConfig(model.tool_id);

  return (
    <article className="rounded-lg border border-border bg-card/40 p-5">
      <header className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">{model.name}</h2>
          <p className="text-xs text-muted-foreground">
            {model.tool_id} · v{model.version}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">{model.description}</p>
        </div>
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-xs",
            model.loaded
              ? "bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/30"
              : "bg-muted text-muted-foreground ring-1 ring-border",
          )}
        >
          {model.loaded ? "загружена" : "не загружена"}
        </span>
      </header>

      {config.isLoading && (
        <p className="text-sm text-muted-foreground">Загрузка настроек…</p>
      )}

      {config.isError && (
        <p
          role="alert"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
        >
          {formatApiError(config.error)}
        </p>
      )}

      {config.data && config.data.schema.length === 0 && (
        <p className="text-sm text-muted-foreground">
          Эта модель не объявляет настраиваемых параметров.
        </p>
      )}

      {config.data && config.data.schema.length > 0 && (
        <SettingsForm toolId={model.tool_id} initial={config.data} />
      )}
    </article>
  );
}

function SettingsForm({
  toolId,
  initial,
}: {
  toolId: string;
  initial: ToolSettingsConfig;
}) {
  const [draft, setDraft] = useState<Record<string, unknown>>(initial.values);
  const patch = usePatchToolConfig(toolId);

  const dirty = initial.schema.some((f) => draft[f.key] !== initial.values[f.key]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Send only the keys whose values differ — preserves "no value set" semantics.
    const changed: Record<string, unknown> = {};
    for (const f of initial.schema) {
      if (draft[f.key] !== initial.values[f.key]) {
        changed[f.key] = draft[f.key];
      }
    }
    if (Object.keys(changed).length > 0) {
      patch.mutate(
        { values: changed },
        { onSuccess: (data) => setDraft(data.values) },
      );
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {initial.schema.map((f) => (
        <FieldRenderer
          key={f.key}
          field={f}
          value={draft[f.key]}
          onChange={(v) => setDraft((d) => ({ ...d, [f.key]: v }))}
        />
      ))}

      <div className="flex items-center gap-3 pt-2">
        <Button
          type="submit"
          disabled={!dirty || patch.isPending}
          className="gap-2"
        >
          {patch.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <Save className="h-4 w-4" aria-hidden />
          )}
          Сохранить
        </Button>
        {patch.isError && (
          <span role="alert" className="text-sm text-destructive">
            {formatApiError(patch.error)}
          </span>
        )}
        {patch.isSuccess && !dirty && (
          <span className="text-sm text-emerald-400">Сохранено</span>
        )}
      </div>
    </form>
  );
}

function FieldRenderer({
  field,
  value,
  onChange,
}: {
  field: SettingField;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (field.type === "select") {
    return <SelectField field={field} value={value as string} onChange={onChange} />;
  }
  if (field.type === "toggle") {
    return <ToggleField field={field} value={value as boolean} onChange={onChange} />;
  }
  if (field.type === "number") {
    return <NumberField field={field} value={value as number} onChange={onChange} />;
  }
  return null;
}

function SelectField({
  field,
  value,
  onChange,
}: {
  field: SettingField;
  value: string;
  onChange: (v: string) => void;
}) {
  const selected = field.options?.find((o) => o.value === value);
  return (
    <fieldset className="space-y-2">
      <legend className="text-sm font-medium">{field.label}</legend>
      {field.description && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
      <div className="grid gap-2 sm:grid-cols-2">
        {field.options?.map((opt) => {
          const active = opt.value === value;
          return (
            <label
              key={opt.value}
              className={cn(
                "flex cursor-pointer flex-col rounded-md border px-3 py-2 transition-colors",
                active
                  ? "border-primary bg-primary/10"
                  : "border-border bg-background hover:bg-secondary/30",
              )}
            >
              <span className="flex items-center gap-2 text-sm font-medium">
                <input
                  type="radio"
                  name={field.key}
                  value={opt.value}
                  checked={active}
                  onChange={() => onChange(opt.value)}
                  className="accent-primary"
                />
                {opt.label}
              </span>
              {opt.description && (
                <span className="ml-5 text-xs text-muted-foreground">
                  {opt.description}
                </span>
              )}
            </label>
          );
        })}
      </div>
      {selected?.description && (
        <p className="sr-only">{selected.description}</p>
      )}
    </fieldset>
  );
}

function ToggleField({
  field,
  value,
  onChange,
}: {
  field: SettingField;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-3">
      <input
        type="checkbox"
        checked={Boolean(value)}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 accent-primary"
      />
      <span className="space-y-0.5">
        <span className="block text-sm font-medium">{field.label}</span>
        {field.description && (
          <span className="block text-xs text-muted-foreground">
            {field.description}
          </span>
        )}
      </span>
    </label>
  );
}

function NumberField({
  field,
  value,
  onChange,
}: {
  field: SettingField;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-sm font-medium">{field.label}</span>
      {field.description && (
        <span className="text-xs text-muted-foreground">{field.description}</span>
      )}
      <input
        type="number"
        value={Number.isFinite(value) ? value : ""}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(n);
        }}
        min={field.min}
        max={field.max}
        step={field.step}
        className="w-32 rounded-md border border-input bg-background px-2 py-1 text-sm"
      />
    </label>
  );
}
