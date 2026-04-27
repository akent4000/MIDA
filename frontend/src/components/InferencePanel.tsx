import { useEffect, useMemo, useState } from "react";
import { Loader2, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ResultCard } from "@/components/ResultCard";
import {
  useExplanationImage,
  useInferenceResult,
  useModels,
  useRunInference,
  useTaskStatus,
  type TaskStatus,
} from "@/hooks/useInference";
import { formatApiError } from "@/lib/errors";

interface Props {
  studyId: string;
  onOverlayChange: (image: HTMLImageElement | null, opacity: number) => void;
}

const STATUS_LABEL: Record<TaskStatus, string> = {
  idle: "Готов",
  queued: "В очереди…",
  running: "Выполняется…",
  done: "Готово",
  failed: "Ошибка",
};

export function InferencePanel({ studyId, onOverlayChange }: Props) {
  const models = useModels();
  const [toolId, setToolId] = useState<string>("");
  const [inference, setInference] = useState<{ id: string; taskId: string } | null>(null);
  const [overlayOn, setOverlayOn] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);

  useEffect(() => {
    if (!toolId && models.data && models.data.length > 0) {
      setToolId(models.data[0].tool_id);
    }
  }, [models.data, toolId]);

  const runMutation = useRunInference();
  const taskState = useTaskStatus(inference?.taskId);
  const isDone = taskState.status === "done";
  const resultQuery = useInferenceResult(inference?.id, isDone);
  const explanation = useExplanationImage(inference?.id, isDone);

  const overlayImage = overlayOn ? explanation.image : null;
  useEffect(() => {
    onOverlayChange(overlayImage, overlayOpacity);
  }, [overlayImage, overlayOpacity, onOverlayChange]);

  const inProgress =
    runMutation.isPending ||
    taskState.status === "queued" ||
    taskState.status === "running";

  const submitError = runMutation.error ? formatApiError(runMutation.error) : null;
  const taskError = taskState.status === "failed" ? taskState.message ?? "Задача завершилась с ошибкой." : null;
  const error = submitError ?? taskError;

  const handleRun = () => {
    if (!toolId) return;
    runMutation.mutate(
      { studyId, toolId },
      {
        onSuccess: (resp) => {
          setInference({ id: resp.inference_id, taskId: resp.task_id });
        },
      },
    );
  };

  const statusTone = useMemo(() => {
    switch (taskState.status) {
      case "running":
      case "queued":
        return "text-blue-300";
      case "done":
        return "text-emerald-300";
      case "failed":
        return "text-destructive";
      default:
        return "text-muted-foreground";
    }
  }, [taskState.status]);

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Инференс
      </h3>

      <label className="flex flex-col gap-1 text-xs">
        <span className="text-muted-foreground">Модель</span>
        <select
          value={toolId}
          onChange={(e) => setToolId(e.target.value)}
          disabled={models.isLoading || inProgress}
          className="rounded-md border border-border bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
          aria-label="Модель"
        >
          {models.isLoading && <option>Загрузка…</option>}
          {models.data?.map((m) => (
            <option key={m.tool_id} value={m.tool_id}>
              {m.name} {m.version ? `(${m.version})` : ""}
            </option>
          ))}
        </select>
      </label>

      <Button
        onClick={handleRun}
        disabled={!toolId || inProgress}
        className="w-full"
        size="sm"
      >
        {inProgress ? (
          <>
            <Loader2 className="mr-1 h-4 w-4 animate-spin" aria-hidden /> Выполняется…
          </>
        ) : (
          <>
            <Play className="mr-1 h-4 w-4" aria-hidden /> Запустить модель
          </>
        )}
      </Button>

      {(inference || runMutation.isPending) && (
        <p className={`text-xs ${statusTone}`} role="status">
          {STATUS_LABEL[taskState.status]}
        </p>
      )}

      {error && (
        <p
          role="alert"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive"
        >
          {error}
        </p>
      )}

      {isDone && resultQuery.data && <ResultCard result={resultQuery.data} />}

      {isDone && (
        <div className="space-y-2 rounded-md border border-border bg-card/40 p-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold uppercase tracking-wide text-muted-foreground">
              Grad-CAM
            </span>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={overlayOn}
                onChange={(e) => setOverlayOn(e.target.checked)}
                disabled={!explanation.image}
                aria-label="Показывать Grad-CAM"
                className="accent-primary"
              />
              <span className="text-muted-foreground">показывать</span>
            </label>
          </div>
          {explanation.isLoading && (
            <p className="text-xs text-muted-foreground">Загрузка тепловой карты…</p>
          )}
          {explanation.error && (
            <p className="text-xs text-destructive">{explanation.error.message}</p>
          )}
          {explanation.image && (
            <label className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Прозрачность</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                className="flex-1 accent-primary"
                aria-label="Прозрачность Grad-CAM"
              />
              <span className="w-10 text-right font-mono tabular-nums">
                {Math.round(overlayOpacity * 100)}%
              </span>
            </label>
          )}
        </div>
      )}
    </div>
  );
}
