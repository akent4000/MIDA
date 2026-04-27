import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useUploadStudy } from "@/hooks/useStudies";
import { formatApiError } from "@/lib/errors";
import { cn } from "@/lib/utils";

const ACCEPTED_EXTENSIONS = [".dcm", ".png", ".jpg", ".jpeg", ".nii", ".nii.gz"];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function UploadPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const upload = useUploadStudy();

  const onDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) setFile(dropped);
  }, []);

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    if (picked) setFile(picked);
  };

  const reset = () => {
    setFile(null);
    upload.reset();
    if (inputRef.current) inputRef.current.value = "";
  };

  const submit = () => {
    if (!file) return;
    upload.mutate(file, {
      onSuccess: (study) => navigate(`/studies/${study.id}`),
    });
  };

  return (
    <section className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Новое исследование</h1>
        <p className="text-sm text-muted-foreground">
          Загрузите DICOM/PNG/NIfTI-файл, чтобы запустить анализ.
        </p>
      </header>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "flex h-64 cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed bg-card/40 px-6 text-center transition-colors",
          isDragging
            ? "border-primary bg-primary/10 text-foreground"
            : "border-border text-muted-foreground hover:border-primary/60 hover:text-foreground",
        )}
        role="button"
        tabIndex={0}
        aria-label="Перетащите файл или нажмите, чтобы выбрать"
      >
        <Upload className="h-8 w-8" aria-hidden />
        <div className="space-y-1">
          <p className="text-sm font-medium">Перетащите файл сюда или нажмите, чтобы выбрать</p>
          <p className="text-xs text-muted-foreground">
            Поддерживаемые форматы: {ACCEPTED_EXTENSIONS.join(", ")}
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(",")}
          className="sr-only"
          onChange={onFileChange}
          data-testid="upload-input"
        />
      </div>

      {file && (
        <div className="flex items-center justify-between rounded-md border border-border bg-card/40 px-4 py-3 text-sm">
          <div className="min-w-0">
            <p className="truncate font-medium">{file.name}</p>
            <p className="text-xs text-muted-foreground">{formatBytes(file.size)}</p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={reset}
              disabled={upload.isPending}
              aria-label="Очистить"
            >
              <X className="h-4 w-4" />
            </Button>
            <Button onClick={submit} disabled={upload.isPending}>
              {upload.isPending ? "Загрузка…" : "Загрузить"}
            </Button>
          </div>
        </div>
      )}

      {upload.isError && (
        <div
          role="alert"
          className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive"
        >
          {formatApiError(upload.error)}
        </div>
      )}
    </section>
  );
}
