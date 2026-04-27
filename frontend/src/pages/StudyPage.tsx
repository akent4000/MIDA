import { useCallback, useState } from "react";
import { useParams } from "react-router-dom";
import { InferencePanel } from "@/components/InferencePanel";
import { MetadataPanel } from "@/components/MetadataPanel";
import { StudyViewer } from "@/components/StudyViewer";
import { useStudy } from "@/hooks/useStudies";
import { useStudyImage } from "@/hooks/useStudyImage";
import { formatApiError } from "@/lib/errors";

export function StudyPage() {
  const { studyId } = useParams<{ studyId: string }>();
  const { data: study, isLoading, isError, error } = useStudy(studyId);
  const { image, isLoading: imageLoading, error: imageError } = useStudyImage(studyId);

  const [overlay, setOverlay] = useState<HTMLImageElement | null>(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);

  const handleOverlayChange = useCallback(
    (img: HTMLImageElement | null, opacity: number) => {
      setOverlay(img);
      setOverlayOpacity(opacity);
    },
    [],
  );

  return (
    <section className="space-y-4">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Исследование</h1>
        <p className="text-xs text-muted-foreground">ID: {studyId}</p>
      </header>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
        <StudyViewer
          image={image}
          isLoading={imageLoading}
          error={imageError ? imageError.message : null}
          overlay={overlay}
          overlayOpacity={overlayOpacity}
        />
        <aside className="space-y-5 rounded-lg border border-border bg-card/40 p-4">
          {studyId && (
            <InferencePanel studyId={studyId} onOverlayChange={handleOverlayChange} />
          )}

          {isLoading && <p className="text-sm text-muted-foreground">Загрузка метаданных…</p>}
          {isError && (
            <p
              role="alert"
              className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
            >
              {formatApiError(error)}
            </p>
          )}
          {study && <MetadataPanel study={study} />}
        </aside>
      </div>
    </section>
  );
}
